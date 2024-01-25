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
