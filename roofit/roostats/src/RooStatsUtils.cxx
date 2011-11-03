// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////
// RooStats
//
// namespace for classes and functions of the RooStats package
/////////////////////////////////////////
#include "Rtypes.h"

#if !defined(R__ALPHA) && !defined(R__SOLARIS) && !defined(R__ACC) && !defined(R__FBSD)
NamespaceImp(RooStats)
#endif


#include "RooProdPdf.h"
#include "RooSimultaneous.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/RooStatsUtils.h"
#include <typeinfo>

// this file is only for the documentation of RooStats namespace

namespace RooStats { 



 void FactorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints) {
   // utility function to factorize constraint terms from a pdf 
   // (from G. Petrucciani)
   const std::type_info & id = typeid(pdf);
   if (id == typeid(RooProdPdf)) {
      RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
      RooArgList list(prod->pdfList());
      for (int i = 0, n = list.getSize(); i < n; ++i) {
         RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            FactorizePdf(observables, *pdfi, obsTerms, constraints);
         }
      } else if (id == typeid(RooSimultaneous) ) {    //|| id == typeid(RooSimultaneousOpt)) {
         RooSimultaneous *sim  = dynamic_cast<RooSimultaneous *>(&pdf);
         RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone();
         for (int ic = 0, nc = cat->numBins((const char *)0); ic < nc; ++ic) {
            cat->setBin(ic);
            FactorizePdf(observables, *sim->getPdf(cat->getLabel()), obsTerms, constraints);
         }
         delete cat;
      } else if (pdf.dependsOn(observables)) {
         if (!obsTerms.contains(pdf)) obsTerms.add(pdf);
      } else {
         if (!constraints.contains(pdf)) constraints.add(pdf);
      }
   }


   void FactorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints) {
      // utility function to factorize constraint terms from a pdf 
      // (from G. Petrucciani)
      return FactorizePdf(*model.GetObservables(), pdf, obsTerms, constraints);
   }


   RooAbsPdf * MakeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name) { 
      // make a nuisance pdf by factorizing out all constraint terms in a common pdf 
      RooArgList obsTerms, constraints;
      FactorizePdf(observables, pdf, obsTerms, constraints);
      return new RooProdPdf(name,"", constraints);
   }

   RooAbsPdf * MakeNuisancePdf(const RooStats::ModelConfig &model, const char *name) { 
      // make a nuisance pdf by factorizing out all constraint terms in a common pdf 
      return MakeNuisancePdf(*model.GetPdf(), *model.GetObservables(), name);
   }






}
