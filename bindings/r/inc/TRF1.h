// @(#)root/r:$Id$
// Author: Omar Zapata   26/05/2014


/*************************************************************************
 * Copyright (C)  2014, Omar Andres Zapata Mesa                          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRF1
#define ROOT_R_TRF1

#ifndef ROOT_TF1
#include<TF1.h>
#endif

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

//________________________________________________________________________________________________________
/**
   This is TF1's wrapper class for R


   @ingroup R
*/

namespace ROOT {
   namespace R {
      class TRF1: public TF1 {
      public:
         TRF1():TF1(){}
         TRF1(const TF1 &f1): TF1(f1) {}
         TRF1(TString name, TString formula, Double_t xmin = 0, Double_t xmax = 1):TF1(name.Data(), formula.Data(), xmin, xmax){}
         std::vector<Double_t> Eval(std::vector<Double_t> x);
         void Draw(){TF1::Draw();}
         void Draw(TString opt){TF1::Draw(opt.Data());}
         inline Int_t Write(const TString name) {
            return TF1::Write(name.Data());
         }
         inline Int_t Write(const TString name, Int_t option,Int_t bufsize) {
            return TF1::Write(name.Data(),option,bufsize);
         }
      };
   }
}

namespace Rcpp {
   template<> SEXP wrap(const TF1 &f)
   {
      return Rcpp::wrap(ROOT::R::TRF1(f));
   }
   template<> TF1 as(SEXP f)
   {
      return Rcpp::as<ROOT::R::TRF1>(f);
   }
}

ROOTR_EXPOSED_CLASS_INTERNAL(TRF1)

//______________________________________________________________________________
std::vector<Double_t> ROOT::R::TRF1::Eval(std::vector<Double_t> x)
{
   std::vector<Double_t> result(x.size());
   for (unsigned int i = 0; i < x.size(); i++) result[i] = TF1::Eval(x[i]);
   return result;
}


ROOTR_MODULE(ROOTR_TRF1)
{

   ROOT::R::class_<ROOT::R::TRF1>("TRF1", "1-Dim ROOT's function class")
   .constructor<TString , TString , Double_t, Double_t>()
   .method("Eval", (std::vector<Double_t> (ROOT::R::TRF1::*)(std::vector<Double_t>))&ROOT::R::TRF1::Eval)
   .method("Eval", (Double_t (ROOT::R::TRF1::*)(Double_t))&ROOT::R::TRF1::Eval)
   .method("Draw", (void (ROOT::R::TRF1::*)())(&ROOT::R::TRF1::Draw))
   .method("Draw", (void (ROOT::R::TRF1::*)(TString))(&ROOT::R::TRF1::Draw))
   .method("SetRange", (void (ROOT::R::TRF1::*)(Double_t, Double_t))(&ROOT::R::TRF1::SetRange))
   .method("SetParameter", (void (ROOT::R::TRF1::*)(Int_t, Double_t))(&ROOT::R::TRF1::SetParameter))
   .method("Write", (Int_t(ROOT::R::TRF1::*)(TString, Int_t, Int_t))(&ROOT::R::TRF1::Write))
   .method("Write", (Int_t(ROOT::R::TRF1::*)(TString))(&ROOT::R::TRF1::Write))
   ;
}
#endif
