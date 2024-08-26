// Author: Enric Tejedor CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPython
#define ROOT_TPython

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPython                                                                  //
//                                                                          //
// Access to the python interpreter and API onto PyROOT.                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

// Bindings
#include "TPyReturn.h"

// ROOT
#include "TObject.h"

#include "ROOT/RConfig.hxx" // R__DEPRECATED

#include <type_traits>

class TPyResult {

public:
   template<class T>
   T Get() const {
      static_assert(std::is_pointer<T>::value, "Expected a pointer");
      return static_cast<T>(fVoidPtr);
   }

   void Set(std::string val) { fString = std::move(val); }
   void Set(Double_t val) { fDouble = val; }
   void Set(Long_t val) { fLong = val; }
   void Set(ULong_t val) { fUnsignedLong = val; }
   void Set(Char_t val) { fChar = val; }
   void Set(void *val) { fVoidPtr = val; }

private:
   std::string fString;
   Double_t fDouble;
   Long_t fLong;
   ULong_t fUnsignedLong;
   Char_t fChar;
   void *fVoidPtr = nullptr;
};

template<> inline std::string TPyResult::Get<std::string>() const { return fString; }
template<> inline Double_t TPyResult::Get<Double_t>() const { return fDouble; }
template<> inline Long_t TPyResult::Get<Long_t>() const { return fLong; }
template<> inline ULong_t TPyResult::Get<ULong_t>() const { return fUnsignedLong; }
template<> inline Char_t TPyResult::Get<Char_t>() const { return fChar; }

////////////////////////////////////////////////////////////////////////////////
/// Get result buffer for the communication between the Python and C++
/// interpreters. Meant to be used in the Python code passed to
/// TPython::Exec().

TPyResult &TPyBuffer();

class TPython {

private:
   static Bool_t Initialize();

public:
   // import a python module, making its classes available
   static Bool_t Import(const char *name);

   // load a python script as if it were a macro
   static void LoadMacro(const char *name);

   // execute a python stand-alone script, with argv CLI arguments
   static void ExecScript(const char *name, int argc = 0, const char **argv = nullptr);

   // execute a python statement (e.g. "import ROOT" )
   static Bool_t Exec(const char *cmd, TPyResult *result = nullptr);

   // evaluate a python expression (e.g. "1+1")
   static const TPyReturn Eval(const char *expr) R__DEPRECATED(6,36, "Use TPython::Exec() In combination with TPython::Result() instead.");

   // bind a ROOT object with, at the python side, the name "label"
   static Bool_t Bind(TObject *object, const char *label);

   // enter an interactive python session (exit with ^D)
   static void Prompt();

   // type verifiers for CPPInstance
   static Bool_t CPPInstance_Check(PyObject *pyobject);
   static Bool_t CPPInstance_CheckExact(PyObject *pyobject);

   // type verifiers for CPPOverload
   static Bool_t CPPOverload_Check(PyObject *pyobject);
   static Bool_t CPPOverload_CheckExact(PyObject *pyobject);

   // CPPInstance to void* conversion
   static void *CPPInstance_AsVoidPtr(PyObject *pyobject);

   // void* to CPPInstance conversion, returns a new reference
   static PyObject *CPPInstance_FromVoidPtr(void *addr, const char *classname, Bool_t python_owns = kFALSE);

   virtual ~TPython() {}

   ClassDef(TPython, 0) // Access to the python interpreter
};

#endif
