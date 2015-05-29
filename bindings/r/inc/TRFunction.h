// @(#)root/r:$Id$
// Author: Omar Zapata   16/06/2013


/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRFunction
#define ROOT_R_TRFunction

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
      class TRFunction: public TObject {
         friend class TRInterface;
         friend SEXP Rcpp::wrap<TRFunction>(const TRFunction &f);
      protected:
         TRInternalFunction *f;
      public:
         TRFunction();

         TRFunction(const TRFunction &fun);
         //________________________________________________________________________________________________________
         template<class T> TRFunction(T fun) {
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

         ClassDef(TRFunction, 0) //
      };
   }
}



#endif
