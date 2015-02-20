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

#ifndef ROOT_TMatrixDSym
#include "TMatrixDSym.h"
#endif

class TFitResult:public TNamed, public ROOT::Fit::FitResult {

public:

   // Default constructor for I/O
   TFitResult(int status = 0): TNamed("TFitResult","TFitResult"),
                           ROOT::Fit::FitResult() {
      fStatus = status;
   };

   // constructor from name and title
   TFitResult(const char * name, const char * title) :
      TNamed(name,title),
      ROOT::Fit::FitResult()
   {}

   // constructor from an FitResult
   TFitResult(const ROOT::Fit::FitResult& f);

   virtual ~TFitResult() {}


   virtual void  Print(Option_t *option="") const;

   TMatrixDSym GetCovarianceMatrix() const;

   TMatrixDSym GetCorrelationMatrix() const;


   using TObject::Error;

   // need to re-implement to solve conflict with TObject::Error
   double Error(unsigned int i) const {
      return ParError(i);
   }

private:
   ClassDef(TFitResult,1)  // Class holding the result of the fit
};

#endif
