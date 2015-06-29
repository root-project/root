// @(#)root/r:$Id$
// Author: Omar Zapata   16/06/2013


/*************************************************************************
 * Copyright (C) 2013-2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRFunctionExport
#define ROOT_R_TRFunctionExport

#ifndef ROOT_R_TRInternalFunction
#include<TRInternalFunction.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a base class to pass functions from ROOT to R


   @ingroup R
*/

namespace ROOT {
   namespace R {

      class TRInterface;
      class TRFunctionExport: public TObject {
         friend class TRInterface;
         friend SEXP Rcpp::wrap<TRFunctionExport>(const TRFunctionExport &f);
      protected:
         TRInternalFunction *f;
      public:
         TRFunctionExport();

         TRFunctionExport(const TRFunctionExport &fun);
         //________________________________________________________________________________________________________
         template<class T> TRFunctionExport(T fun) {
            //template constructor that supports a lot
            // of function's prototypes
            f = new TRInternalFunction(fun);
         }

         //________________________________________________________________________________________________________
         template<class T> void SetFunction(T fun) {
            //template method that supports a lot
            // of function's prototypes
            f = new TRInternalFunction(fun);
         }

         ClassDef(TRFunctionExport, 0) //
      };
   }
}



#endif
