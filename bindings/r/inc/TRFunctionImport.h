// @(#)root/r:$Id$
// Author: Omar Zapata   28/06/2015


/*************************************************************************
 * Copyright (C) 2015, Omar Andres Zapata Mesa                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRFunctionImport
#define ROOT_R_TRFunctionImport


#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

#ifndef ROOT_R_TRObject
#include<TRObject.h>
#endif

#ifndef Rcpp_hpp
#include <Rcpp.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a base class to pass functions from ROOT to R


   @ingroup R
*/

namespace ROOT {
   namespace R {

      class TRInterface;
      class TRFunctionImport: public TObject {
         friend class TRInterface;
         friend SEXP Rcpp::wrap<TRFunctionImport>(const TRFunctionImport &f);
         friend TRFunctionImport Rcpp::as<>(SEXP);

      protected:
         Rcpp::Function *f;
         TRFunctionImport(const Rcpp::Function &fun){
             *f=fun;
         }
         
      public:
        TRFunctionImport(const TString& name);
        TRFunctionImport(const TString& name, const TString& ns);
        TRFunctionImport(const TRFunctionImport &fun);
        TRFunctionImport(SEXP obj);
        
        ~TRFunctionImport(){if(f) delete f;}
        SEXP operator()(){return (*f)();}
        #include<TRFunctionImport__oprtr.h>
         ClassDef(TRFunctionImport, 0) //
      };
   }
}



#endif
