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

#ifndef ROOT_R_TRObjectProxy
#include<TRObjectProxy.h>
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
      protected:
         Rcpp::Function *f;
         TString fFunctionName;
         TString fNameSpace;
         
      public:
        TRFunctionImport(const TString& name);
        TRFunctionImport(const TString& name, const TString& ns);
        TRFunctionImport(const TRFunctionImport &fun);
        ~TRFunctionImport(){if(f) delete f;}
        TRObjectProxy operator()(){return (*f)();}
        #include<TRFunctionImport__oprtr.h>
         ClassDef(TRFunctionImport, 0) //
      };
   }
}



#endif
