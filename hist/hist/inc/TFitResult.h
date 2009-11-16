// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline Tue Nov 10 15:01:24 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitResult
#define ROOT_TFitResult

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitResult                                                           //
//                                                                      //
// Provides a way to view the fit result and to store them.             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_FIT_FitResult
#include "Fit/FitResult.h"
#endif

#include "TMatrixDSym.h"

class TFitResult:public TNamed, public ROOT::Fit::FitResult {
public:

   // Default constructor for I/O
   TFitResult(int status = 0): TNamed("TFitResult","TFitResult"), 
                           ROOT::Fit::FitResult() {
      fStatus = status;
   };

   // constructor from an IFitResult
   TFitResult(const ROOT::Fit::FitResult& f): TNamed("TFitResult","TFitResult"),
                                              ROOT::Fit::FitResult(f) {};

   virtual ~TFitResult() {};

   // method using TMatrix 
   TMatrixDSym  GetCovarianceMatrix() const;

   ClassDef(TFitResult,1)  // Class holding the result of the fit 
};

#endif
