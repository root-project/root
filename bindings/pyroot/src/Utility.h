// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

#ifndef PYROOT_UTILITY_H
#define PYROOT_UTILITY_H

// ROOT
#include "DllImport.h"
#include "TClass.h"

// Standard
#include <map>
#include <string>
#include <vector>


namespace PyROOT {

   class PyCallable;

   R__EXTERN dict_lookup_func gDictLookupOrg;
   R__EXTERN Bool_t gDictLookupActive;

// general place holder for function parameters
   union TParameter_t {
      Long_t     fLong;
      ULong_t    fULong;
      Long64_t   fLongLong;
      ULong64_t  fULongLong;
      Double_t   fDouble;
      void*      fVoidp;
   };

// additional converter functions
   ULong_t PyLongOrInt_AsULong( PyObject* pyobject );
   ULong64_t PyLongOrInt_AsULong64( PyObject* pyobject );

   namespace Utility {

   // convenience functions for adding methods to classes
      Bool_t AddToClass( PyObject* pyclass, const char* label, PyCFunction cfunc,
                         int flags = METH_VARARGS );
      Bool_t AddToClass( PyObject* pyclass, const char* label, const char* func );
      Bool_t AddToClass( PyObject* pyclass, const char* label, PyCallable* pyfunc );

      Bool_t AddUsingToClass( PyObject* pyclass, const char* method );

   // helpers for dynamically constructing binary operators
      Bool_t AddBinaryOperator( PyObject* left, PyObject* right, const char* op, const char* label = "" );
      Bool_t AddBinaryOperator( PyObject* pyclass, const char* op, const char* label = "" );
      Bool_t AddBinaryOperator( PyObject* pyclass, const std::string& lcname, const std::string& rcname,
                                const char* op, const char* label = "" );

   // helper for template classes and methods
      PyObject* BuildTemplateName( PyObject* pyname, PyObject* args, int argoff );

   // initialize proxy type objects
      Bool_t InitProxy( PyObject* module, PyTypeObject* pytype, const char* name );

   // retrieve the memory buffer from pyobject, return buflength, tc (optional) is python
   // array.array type code, size is type size, buf will point to buffer, and if check is
   // true, some heuristics will be applied to check buffer compatibility with the type
      int GetBuffer( PyObject* pyobject, char tc, int size, void*& buf, Bool_t check = kTRUE );

   // memory handling
      enum EMemoryPolicy { kHeuristics = 1, kStrict = 2 };
      R__EXTERN EMemoryPolicy gMemoryPolicy;
      Bool_t SetMemoryPolicy( EMemoryPolicy e );

   // signal safety
      enum ESignalPolicy { kFast = 1, kSafe = 2 };
      R__EXTERN ESignalPolicy gSignalPolicy;
      Bool_t SetSignalPolicy( ESignalPolicy e );

   // data/operator mappings
      std::string MapOperatorName( const std::string& name, Bool_t bTakesParames );

   // meta information
      const std::string Compound( const std::string& name );
      const std::string ClassName( PyObject* pyobj );
      const std::string ResolveTypedef( const std::string& name, TClass* containing_scope = 0 );

   // the inconsistent interface of the next call is to be fixed once #100740 resolves
      Long_t GetObjectOffset( ClassInfo_t* clCurrent, ClassInfo_t* clDesired, void* obj );

   // CINT integration
      void ErrMsgCallback( char* msg );
      void ErrMsgHandler( int level, Bool_t abort, const char* location, const char* msg );

   // create a wrapper for the python function with Cling; returns function pointer
      void* CreateWrapperMethod( PyObject* pyfunc, Long_t user,
         const char* retType, const std::vector<std::string>& signature, const char* callback );

   // for threading: save call to PyErr_Occurred()
      PyObject* PyErr_Occurred_WithGIL();

   // for GUI driving; install/remove event callback
      PyObject* InstallGUIEventInputHook();
      PyObject* RemoveGUIEventInputHook();

   } // namespace Utility

} // namespace PyROOT

#endif // !PYROOT_UTILITY_H
