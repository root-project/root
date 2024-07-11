// Author: Enric Tejedor CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL
//
// /*************************************************************************
//  * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
//  * All rights reserved.                                                  *
//  *                                                                       *
//  * For the licensing terms see $ROOTSYS/LICENSE.                         *
//  * For the list of contributors see $ROOTSYS/README/CREDITS.             *
//  *************************************************************************/

// Bindings
#include "CPyCppyy/API.h"
#include "TPyReturn.h"

// ROOT
#include "TObject.h"
#include "TInterpreter.h"

// Standard
#include <stdexcept>

//______________________________________________________________________________
//                        Python expression eval result
//                        =============================
//
// Transport class for bringing objects from python (dynamically typed) to Cling
// (statically typed). It is best to immediately cast a TPyReturn to the real
// type, either implicitly (for builtin types) or explicitly (through a void*
// cast for pointers to ROOT objects).
//
// Examples:
//
//  root [0] TBrowser* b = (void*)TPython::Eval( "ROOT.TBrowser()" );
//  root [1] int i = TPython::Eval( "1+1" );
//  root [2] i
//  (int)2
//  root [3] double d = TPython::Eval( "1+3.1415" );
//  root [4] d
//  (double)4.14150000000000063e+00

//- data ---------------------------------------------------------------------
ClassImp(TPyReturn);

//- constructors/destructor --------------------------------------------------
TPyReturn::TPyReturn() : fPyResult{new CPyCppyy::PyResult{}} {}

////////////////////////////////////////////////////////////////////////////////
/// Construct a TPyReturn from a python object. The python object may represent
/// a ROOT object. Steals reference to given python object.

TPyReturn::TPyReturn(PyObject *pyobject) : fPyResult{new CPyCppyy::PyResult{pyobject}} {}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Reference counting for the held python object is in effect.

TPyReturn::~TPyReturn() {}

//- public members -----------------------------------------------------------
TPyReturn::operator char *() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C-style string (may fail).

TPyReturn::operator const char *() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ char (may fail).

TPyReturn::operator Char_t() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ long (may fail).

TPyReturn::operator Long_t() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ unsigned long (may fail).

TPyReturn::operator ULong_t() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to C++ double (may fail).

TPyReturn::operator Double_t() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Cast python return value to ROOT object with dictionary (may fail; note that
/// you have to use the void* converter, as CINT will not call any other).

TPyReturn::operator void *() const
{
   return *fPyResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Direct return of the held PyObject; note the new reference.

TPyReturn::operator PyObject *() const
{
   return *fPyResult;
}
