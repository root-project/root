// @(#)root/r:$Id$
// Author: Omar Zapata   26/05/2014


/*************************************************************************
 * Copyright (C)  2014, Omar Andres Zapata Mesa                          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRGraph
#define ROOT_R_TRGraph

#ifndef ROOT_TGraph
#include<TGraph.h>
#endif

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

//________________________________________________________________________________________________________
/**
   This is TGraph class for R


   @ingroup R
*/

namespace ROOT {
   namespace R {

      class TRGraph: public TGraph {
      public:
         TRGraph(): TGraph() {};
         TRGraph(const TGraph &g): TGraph(g) {};
         TRGraph(Int_t n, std::vector<double> x, std::vector<double> y): TGraph(n, x.data(), y.data()){}
         void Draw(){TGraph::Draw();}
         void Draw(TString opt){TGraph::Draw(opt.Data());}
      };
   }
}

namespace Rcpp {
   template<> SEXP wrap(const TGraph &f)
   {
      return Rcpp::wrap(ROOT::R::TRGraph(f));
   }
   template<> TGraph as(SEXP f)
   {
      return Rcpp::as<ROOT::R::TRGraph>(f);
   }
}

ROOTR_EXPOSED_CLASS_INTERNAL(TRGraph)

ROOTR_MODULE(ROOTR_TRGraph)
{
   ROOT::R::class_<ROOT::R::TRGraph>("TRGraph")
   .constructor<int, std::vector<Double_t>, std::vector<Double_t> >()
   .method("Draw", (void (ROOT::R::TRGraph::*)())(&ROOT::R::TRGraph::Draw))
   .method("Draw", (void (ROOT::R::TRGraph::*)(TString))(&ROOT::R::TRGraph::Draw))
   ;
}
#endif
