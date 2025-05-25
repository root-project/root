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
#include "../../cppyy/CPyCppyy/src/Dimensions.h"

#include "CPyCppyy/API.h"

#include "PyROOTPythonize.h"

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

#include <algorithm>
#include <sstream>

namespace {

// Get the TClass of the C++ object proxied by pyobj
TClass *GetTClass(const PyObject *pyobj)
{
   return TClass::GetClass(Cppyy::GetScopedFinalName(((CPyCppyy::CPPInstance *)pyobj)->ObjectIsA()).c_str());
}

} // namespace

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

static std::pair<void *, std::string> ResolveBranch(TTree *tree, const char *name, TBranch *branch)
{
   // for partial return of a split object
   if (branch->InheritsFrom(TBranchElement::Class())) {
      TBranchElement *be = (TBranchElement *)branch;
      if (be->GetCurrentClass() && (be->GetCurrentClass() != be->GetTargetClass()) && (0 <= be->GetID())) {
         Long_t offset = ((TStreamerElement *)be->GetInfo()->GetElements()->At(be->GetID()))->GetOffset();
         return {be->GetObject() + offset, be->GetCurrentClass()->GetName()};
      }
   }

   // for return of a full object
   if (branch->IsA() == TBranchElement::Class() || branch->IsA() == TBranchObject::Class()) {
      if (branch->GetAddress())
         return {*(void **)branch->GetAddress(), branch->GetClassName()};

      // try leaf, otherwise indicate failure by returning a typed null-object
      TObjArray *leaves = branch->GetListOfLeaves();
      if (!tree->GetLeaf(name) && !(leaves->GetSize() && (leaves->First() == leaves->Last())))
         return {nullptr, branch->GetClassName()};
   }

   return {nullptr, ""};
}

/**
 * @brief Extracts static dimensions from the title of a TLeaf object.
 *
 * The function assumes that the title of the TLeaf object contains dimensions
 * in the format `[dim1][dim2]...`.
 *
 * @note In the current implementation of TLeaf, there is no way to extract the
 *       dimensions without string parsing.
 *
 * @param leaf Pointer to the TLeaf object from which to extract dimensions.
 * @return std::vector<dim_t> A vector containing the extracted dimensions.
 */
static std::vector<dim_t> getMultiDims(std::string const &title)
{
   std::vector<dim_t> dims;
   std::stringstream ss{title};

   while (ss.good()) {
      std::string substr;
      getline(ss, substr, '[');
      getline(ss, substr, ']');
      if (!substr.empty()) {
         dims.push_back(std::stoi(substr));
      }
   }

   return dims;
}

static PyObject *WrapLeaf(TLeaf *leaf)
{
   if (1 < leaf->GetLenStatic() || leaf->GetLeafCount()) {
      bool isStatic = 1 < leaf->GetLenStatic();
      // array types
      std::string typeName = leaf->GetTypeName();
      std::vector<dim_t> dimsVec{leaf->GetNdata()};
      std::string title = leaf->GetTitle();
      // Multidimensional array case
      if (std::count(title.begin(), title.end(), '[') >= 2) {
         dimsVec = getMultiDims(title);
      }
      CPyCppyy::Dimensions dims{static_cast<dim_t>(dimsVec.size()), dimsVec.data()};
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

// Allow access to branches/leaves as if they were data members Returns a
// Python tuple where the first element is either the desired CPyCppyy proxy,
// or an address that still needs to be wrapped by the caller in a proxy using
// cppyy.ll.cast. In the latter case, the second tuple element is the target
// type name. Otherwise, the second element is an empty string.
PyObject *PyROOT::GetBranchAttr(PyObject * /*self*/, PyObject *args)
{
   PyObject *self = nullptr;
   PyObject *pyname = nullptr;

   PyArg_ParseTuple(args, "OU:GetBranchAttr", &self, &pyname);

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
      const auto [finalAddressVoidPtr, finalTypeName] = ResolveBranch(tree, name, branch);
      if (!finalTypeName.empty()) {
         PyObject *outTuple = PyTuple_New(2);
         PyTuple_SET_ITEM(outTuple, 0, PyLong_FromLongLong((intptr_t)finalAddressVoidPtr));
         PyTuple_SET_ITEM(outTuple, 1, CPyCppyy_PyText_FromString((finalTypeName + "*").c_str()));
         return outTuple;
      }
   }

   // if not, try leaf
   if (TLeaf *leaf = SearchForLeaf(tree, name, branch)) {
      // found a leaf, extract value and wrap with a Python object according to its type
      auto wrapper = WrapLeaf(leaf);
      if (wrapper != nullptr) {
         PyObject *outTuple = PyTuple_New(2);
         PyTuple_SET_ITEM(outTuple, 0, wrapper);
         PyTuple_SET_ITEM(outTuple, 1, CPyCppyy_PyText_FromString(""));
         return outTuple;
      }
   }

   // confused
   PyErr_Format(PyExc_AttributeError, "\'%s\' object has no attribute \'%s\'", tree->IsA()->GetName(), name);
   return 0;
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

   if (PyArg_ParseTuple(args, "OO!OO!|O!:Branch", &treeObj, &PyUnicode_Type, &name, &address, &PyUnicode_Type,
                        &leaflist, &PyInt_Type, &bufsize)) {

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
            branch = tree->Branch(PyUnicode_AsUTF8(name), buf, PyUnicode_AsUTF8(leaflist), PyInt_AS_LONG(bufsize));
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
   if (PyArg_ParseTuple(args, "OO!O!O|O!O!:Branch", &treeObj, &PyUnicode_Type, &name, &PyUnicode_Type, &clName,
                        &address, &PyInt_Type, &bufsize, &PyInt_Type, &splitlevel)) {
      bIsMatch = true;
   } else {
      PyErr_Clear();
      if (PyArg_ParseTuple(args, "OO!O|O!O!", &treeObj, &PyUnicode_Type, &name, &address, &PyInt_Type, &bufsize,
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
