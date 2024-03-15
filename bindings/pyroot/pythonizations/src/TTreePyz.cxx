// Author: Enric Tejedor CERN  06/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Bindings
#include "../../cppyy/CPyCppyy/src/CPyCppyy.h"
#include "../../cppyy/CPyCppyy/src/CPPInstance.h"
#include "../../cppyy/CPyCppyy/src/ProxyWrappers.h"
#include "../../cppyy/CPyCppyy/src/Utility.h"

// Th API changed a bit with the new CPyCppyy, which we can detect by checking
// if CPYCPPYY_VERSION_HEX is defined.
#ifdef CPYCPPYY_VERSION_HEX
#include "../../cppyy/CPyCppyy/src/Dimensions.h"
#endif

#include "CPyCppyy/API.h"

#include "PyROOTPythonize.h"
#include "PyzCppHelpers.hxx"

// ROOT
#include "TClass.h"
#include "TTree.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TLeaf.h"
#include "TLeafElement.h"
#include "TLeafObject.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"

using namespace CPyCppyy;

static TBranch *SearchForBranch(TTree *tree, const char *name)
{
   TBranch *branch = tree->GetBranch(name);
   if (!branch) {
      // for benefit of naming of sub-branches, the actual name may have a trailing '.'
      branch = tree->GetBranch((std::string(name) + '.').c_str());
   }
   return branch;
}

static TLeaf *SearchForLeaf(TTree *tree, const char *name, TBranch *branch)
{
   TLeaf *leaf = tree->GetLeaf(name);
   if (branch && !leaf) {
      leaf = branch->GetLeaf(name);
      if (!leaf) {
         TObjArray *leaves = branch->GetListOfLeaves();
         if (leaves->GetSize() && (leaves->First() == leaves->Last())) {
            // i.e., if unambiguously only this one
            leaf = (TLeaf *)leaves->At(0);
         }
      }
   }
   return leaf;
}

static PyObject *BindBranchToProxy(TTree *tree, const char *name, TBranch *branch)
{
   // for partial return of a split object
   if (branch->InheritsFrom(TBranchElement::Class())) {
      TBranchElement *be = (TBranchElement *)branch;
      if (be->GetCurrentClass() && (be->GetCurrentClass() != be->GetTargetClass()) && (0 <= be->GetID())) {
         Long_t offset = ((TStreamerElement *)be->GetInfo()->GetElements()->At(be->GetID()))->GetOffset();
         return CPyCppyy::Instance_FromVoidPtr(be->GetObject() + offset, be->GetCurrentClass()->GetName());
      }
   }

   // for return of a full object
   if (branch->IsA() == TBranchElement::Class() || branch->IsA() == TBranchObject::Class()) {
      TClass *klass = TClass::GetClass(branch->GetClassName());
      if (klass && branch->GetAddress())
         return CPyCppyy::Instance_FromVoidPtr(*(void **)branch->GetAddress(), branch->GetClassName());

      // try leaf, otherwise indicate failure by returning a typed null-object
      TObjArray *leaves = branch->GetListOfLeaves();
      if (klass && !tree->GetLeaf(name) && !(leaves->GetSize() && (leaves->First() == leaves->Last())))
         return CPyCppyy::Instance_FromVoidPtr(nullptr, branch->GetClassName());
   }

   return nullptr;
}

static PyObject *WrapLeaf(TLeaf *leaf)
{
   if (1 < leaf->GetLenStatic() || leaf->GetLeafCount()) {
      bool isStatic = 1 < leaf->GetLenStatic();
      // array types
      std::string typeName = leaf->GetTypeName();
#ifdef CPYCPPYY_VERSION_HEX
      dim_t dimsArr[]{ leaf->GetNdata() };
      CPyCppyy::Dimensions dims{1, dimsArr};
#else
      dim_t dims[]{ 1, leaf->GetNdata() }; // first entry is the number of dims
#endif
      Converter *pcnv = CreateConverter(typeName + (isStatic ? "[]" : "*"), dims);

      void *address = 0;
      if (leaf->GetBranch())
         address = (void *)leaf->GetBranch()->GetAddress();
      if (!address)
         address = (void *)leaf->GetValuePointer();

      PyObject *value = pcnv->FromMemory(&address);
      CPyCppyy::DestroyConverter(pcnv);

      return value;
   } else if (leaf->GetValuePointer()) {
      // value types
      Converter *pcnv = CreateConverter(leaf->GetTypeName());
      PyObject *value = 0;
      if (leaf->IsA() == TLeafElement::Class() || leaf->IsA() == TLeafObject::Class())
         value = pcnv->FromMemory((void *)*(void **)leaf->GetValuePointer());
      else
         value = pcnv->FromMemory((void *)leaf->GetValuePointer());
      CPyCppyy::DestroyConverter(pcnv);

      return value;
   }

   return nullptr;
}

// Allow access to branches/leaves as if they were data members
PyObject *GetAttr(PyObject *self, PyObject *pyname)
{
   const char *name_possibly_alias = PyUnicode_AsUTF8(pyname);
   if (!name_possibly_alias)
      return 0;

   // get hold of actual tree
   auto tree = (TTree *)GetTClass(self)->DynamicCast(TTree::Class(), CPyCppyy::Instance_AsVoidPtr(self));

   if (!tree) {
      PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
      return 0;
   }

   // deal with possible aliasing
   const char *name = tree->GetAlias(name_possibly_alias);
   if (!name)
      name = name_possibly_alias;

   // search for branch first (typical for objects)
   TBranch *branch = SearchForBranch(tree, name);

   if (branch) {
      // found a branched object, wrap its address for the object it represents
      auto proxy = BindBranchToProxy(tree, name, branch);
      if (proxy != nullptr)
         return proxy;
   }

   // if not, try leaf
   TLeaf *leaf = SearchForLeaf(tree, name, branch);

   if (leaf) {
      // found a leaf, extract value and wrap with a Python object according to its type
      auto wrapper = WrapLeaf(leaf);
      if (wrapper != nullptr)
         return wrapper;
   }

   // confused
   PyErr_Format(PyExc_AttributeError, "\'%s\' object has no attribute \'%s\'", tree->IsA()->GetName(), name);
   return 0;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Allow branches to be accessed as attributes of a tree.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// Allow access to branches/leaves as if they were Python data attributes of the tree
/// (e.g. mytree.branch)
PyObject *PyROOT::AddBranchAttrSyntax(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__getattr__", (PyCFunction)GetAttr, METH_O);
   Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonization for TTree::SetBranchAddress.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// Modify the behaviour of SetBranchAddress so that proxy references can be passed
/// as arguments from the Python side, more precisely in cases where the C++
/// implementation of the method expects the address of a pointer.
///
/// For example:
/// ~~~{.py}
/// v = ROOT.std.vector('int')()
/// t.SetBranchAddress("my_vector_branch", v)
/// ~~~
PyObject *PyROOT::SetBranchAddressPyz(PyObject * /* self */, PyObject *args)
{
   PyObject *treeObj = nullptr, *name = nullptr, *address = nullptr;

   int argc = PyTuple_GET_SIZE(args);

// Look for the (const char*, void*) overload
   auto argParseStr = "OUO:SetBranchAddress";
   if (argc == 3 && PyArg_ParseTuple(args, argParseStr, &treeObj, &name, &address)) {

      auto tree = (TTree *)GetTClass(treeObj)->DynamicCast(TTree::Class(), CPyCppyy::Instance_AsVoidPtr(treeObj));

      if (!tree) {
         PyErr_SetString(PyExc_TypeError,
                         "TTree::SetBranchAddress must be called with a TTree instance as first argument");
         return nullptr;
      }

      auto branchName = PyUnicode_AsUTF8(name);
      auto branch = tree->GetBranch(branchName);
      if (!branch) {
         PyErr_SetString(PyExc_TypeError, "TTree::SetBranchAddress must be called with a valid branch name");
         return nullptr;
      }

      bool isLeafList = branch->IsA() == TBranch::Class();

      void *buf = 0;
      if (CPyCppyy::Instance_Check(address)) {
         ((CPPInstance *)address)->GetDatamemberCache(); // force creation of cache

         if (((CPPInstance *)address)->fFlags & CPPInstance::kIsReference || isLeafList)
            buf = CPyCppyy::Instance_AsVoidPtr(address);
         else
            buf = (void *)&(((CPPInstance *)address)->GetObjectRaw());
      } else
         Utility::GetBuffer(address, '*', 1, buf, false);

      if (buf != nullptr) {
         auto res = tree->SetBranchAddress(PyUnicode_AsUTF8(name), buf);
         return PyInt_FromLong(res);
      }
   }

   // Not the overload we wanted to pythonize, return None
   Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////
/// Try to match the arguments of TTree::Branch to the following overload:
/// - ( const char*, void*, const char*, Int_t = 32000 )
/// If the match succeeds, invoke Branch on the C++ tree with the right
/// arguments.
PyObject *TryBranchLeafListOverload(int argc, PyObject *args)
{
   PyObject *treeObj = nullptr;
   PyObject *name = nullptr, *address = nullptr, *leaflist = nullptr, *bufsize = nullptr;

   if (PyArg_ParseTuple(args, "OO!OO!|O!:Branch",
                        &treeObj,
                        &PyUnicode_Type, &name,
                        &address,
                        &PyUnicode_Type, &leaflist,
                        &PyInt_Type, &bufsize)) {

      auto tree = (TTree *)GetTClass(treeObj)->DynamicCast(TTree::Class(), CPyCppyy::Instance_AsVoidPtr(treeObj));
      if (!tree) {
         PyErr_SetString(PyExc_TypeError, "TTree::Branch must be called with a TTree instance as first argument");
         return nullptr;
      }

      void *buf = nullptr;
      if (CPPInstance_Check(address))
         buf = CPyCppyy::Instance_AsVoidPtr(address);
      else
         Utility::GetBuffer(address, '*', 1, buf, false);

      if (buf) {
         TBranch *branch = nullptr;
         if (argc == 5) {
            branch = tree->Branch(PyUnicode_AsUTF8(name), buf, PyUnicode_AsUTF8(leaflist),
                                  PyInt_AS_LONG(bufsize));
         } else {
            branch = tree->Branch(PyUnicode_AsUTF8(name), buf, PyUnicode_AsUTF8(leaflist));
         }

         return BindCppObject(branch, Cppyy::GetScope("TBranch"));
      }
   }
   PyErr_Clear();

   Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////
/// Try to match the arguments of TTree::Branch to one of the following
/// overloads:
/// - ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
/// - ( const char*,              T**, Int_t = 32000, Int_t = 99 )
/// If the match succeeds, invoke Branch on the C++ tree with the right
/// arguments.
PyObject *TryBranchPtrToPtrOverloads(int argc, PyObject *args)
{
   PyObject *treeObj = nullptr;
   PyObject *name = nullptr, *clName = nullptr, *address = nullptr, *bufsize = nullptr, *splitlevel = nullptr;

   auto bIsMatch = false;
   if (PyArg_ParseTuple(args, "OO!O!O|O!O!:Branch",
                        &treeObj,
                        &PyUnicode_Type, &name,
                        &PyUnicode_Type, &clName,
                        &address,
                        &PyInt_Type, &bufsize,
                        &PyInt_Type, &splitlevel)) {
      bIsMatch = true;
   } else {
      PyErr_Clear();
      if (PyArg_ParseTuple(args, "OO!O|O!O!",
                           &treeObj,
                           &PyUnicode_Type, &name,
                           &address,
                           &PyInt_Type, &bufsize,
                           &PyInt_Type, &splitlevel)) {
         bIsMatch = true;
      } else {
         PyErr_Clear();
      }
   }

   if (bIsMatch) {
      auto tree = (TTree *)GetTClass(treeObj)->DynamicCast(TTree::Class(), CPyCppyy::Instance_AsVoidPtr(treeObj));
      if (!tree) {
         PyErr_SetString(PyExc_TypeError, "TTree::Branch must be called with a TTree instance as first argument");
         return nullptr;
      }

      std::string klName = clName ? PyUnicode_AsUTF8(clName) : "";
      void *buf = nullptr;

      if (CPPInstance_Check(address)) {
         if (((CPPInstance *)address)->fFlags & CPPInstance::kIsReference)
            buf = (void *)((CPPInstance *)address)->fObject;
         else
            buf = (void *)&((CPPInstance *)address)->fObject;

         if (!clName) {
            klName = GetTClass(address)->GetName();
            argc += 1;
         }
      } else {
         Utility::GetBuffer(address, '*', 1, buf, false);
      }

      if (buf && !klName.empty()) {
         TBranch *branch = nullptr;
         if (argc == 4) {
            branch = tree->Branch(PyUnicode_AsUTF8(name), klName.c_str(), buf);
         } else if (argc == 5) {
            branch = tree->Branch(PyUnicode_AsUTF8(name), klName.c_str(), buf, PyInt_AS_LONG(bufsize));
         } else if (argc == 6) {
            branch = tree->Branch(PyUnicode_AsUTF8(name), klName.c_str(), buf, PyInt_AS_LONG(bufsize),
                                  PyInt_AS_LONG(splitlevel));
         }

         return BindCppObject(branch, Cppyy::GetScope("TBranch"));
      }
   }

   Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonization for TTree::Branch.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// Modify the behaviour of Branch so that proxy references can be passed
/// as arguments from the Python side, more precisely in cases where the C++
/// implementation of the method expects the address of a pointer.
///
/// For example:
/// ~~~{.py}
/// v = ROOT.std.vector('int')()
/// t.Branch('my_vector_branch', v)
/// ~~~
///
/// The following signatures are treated in this pythonization:
/// - ( const char*, void*, const char*, Int_t = 32000 )
/// - ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
/// - ( const char*, T**, Int_t = 32000, Int_t = 99 )
PyObject *PyROOT::BranchPyz(PyObject * /* self */, PyObject *args)
{
   int argc = PyTuple_GET_SIZE(args);

   if (argc >= 3) { // We count the TTree proxy object too
      auto branch = TryBranchLeafListOverload(argc, args);
      if (branch != Py_None)
         return branch;

      branch = TryBranchPtrToPtrOverloads(argc, args);
      if (branch != Py_None)
         return branch;
   }

   // Not the overload we wanted to pythonize, return None
   Py_RETURN_NONE;
}
