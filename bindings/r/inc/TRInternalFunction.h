// @(#)root/r:$Id$
// Author: Omar Zapata   07/06/2014


/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRInternalFunction
#define ROOT_R_TRInternalFunction

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

#ifndef Rcpp_hpp
#include <Rcpp.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a class to support deprecated method to pass function to R's Environment,
   based in Rcpp::InternalFunction


   @ingroup R
*/


namespace Rcpp {
   RCPP_API_CLASS(TRInternalFunction_Impl)
   {
public:

      RCPP_GENERATE_CTOR_ASSIGN(TRInternalFunction_Impl)

#include <TRInternalFunction__ctors.h>
      void update(SEXP) {}
private:

      inline void set(SEXP xp) {
         Rcpp::Environment RCPP = Rcpp::Environment::Rcpp_namespace() ;
         Rcpp::Function intf = RCPP["internal_function"] ;
         Storage::set__(intf(xp)) ;
      }

   };


}


namespace ROOT {
   namespace R {

      typedef Rcpp::TRInternalFunction_Impl<Rcpp::PreserveStorage> TRInternalFunction ;
   }
}

#endif
