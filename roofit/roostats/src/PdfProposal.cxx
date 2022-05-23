// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::PdfProposal
    \ingroup Roostats

PdfProposal is a concrete implementation of the ProposalFunction interface.
It proposes points across the parameter space in the distribution of the
given PDF.

To make Propose(xPrime, x) dependent on x, configure with
PdfProposal::AddMapping(varToUpdate, valueToUse).  For example, suppose we have:

~~~{.cpp}
// our parameter
RooRealVar p("p", "p", 5, 0, 10);

// create mean and sigma for gaussian proposal function
RooRealVar meanP("meanP", "meanP", 0, 10);
RooRealVar sigma("sigma", "sigma", 1, 0, 5);
RooGaussian pGaussian("pGaussian", "pGaussian", p, meanP, sigma);

// configure proposal function
PdfProposal pdfProposal(pGaussian);
pdfProposal.AddMapping(meanP, p); // each call of Propose(xPrime, x), meanP in
                                  // the proposal function will be updated to
                                  // the value of p in x.  this will center the
                                  // proposal function about x's p when
                                  // proposing for xPrime

// To improve performance, PdfProposal has the ability to cache a specified
// number of proposals. If you don't call this function, the default cache size
// is 1, which can be slow.
pdfProposal.SetCacheSize(desiredCacheSize);
~~~

PdfProposal currently uses a fixed cache size. Adaptive caching methods are in the works
for future versions.
*/


#include "Rtypes.h"

#include "RooStats/PdfProposal.h"
#include "RooStats/RooStatsUtils.h"
#include "RooArgSet.h"
#include "RooDataSet.h"
#include "RooAbsPdf.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"

#include <map>

ClassImp(RooStats::PdfProposal);

using namespace RooFit;
using namespace RooStats;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
/// By default, PdfProposal does NOT own the PDF that serves as the
/// proposal density function

PdfProposal::PdfProposal() : ProposalFunction()
{
   fPdf = nullptr;
   fOwnsPdf = false;
   fCacheSize = 1;
   fCachePosition = 0;
   fCache = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// By default, PdfProposal does NOT own the PDF that serves as the
/// proposal density function

PdfProposal::PdfProposal(RooAbsPdf& pdf) : ProposalFunction()
{
   fPdf = &pdf;
   fOwnsPdf = false;
   fCacheSize = 1;
   fCachePosition = 0;
   fCache = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// determine whether these two RooArgSets represent the same point

bool PdfProposal::Equals(RooArgSet& x1, RooArgSet& x2)
{
   if (x1.equals(x2)) {
      for (auto const *r : static_range_cast<RooRealVar*>(x1))
         if (r->getVal() != x2.getRealValue(r->GetName())) {
            return false;
         }
      return true;
   }
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Populate xPrime with a new proposed point

void PdfProposal::Propose(RooArgSet& xPrime, RooArgSet& x)
{
   if (fLastX.empty()) {
      // fLastX not yet initialized
      fLastX.addClone(x);
      // generate initial cache
      RooStats::SetParameters(&x, &fMaster);
      if (fMap.size() > 0) {
         for (fIt = fMap.begin(); fIt != fMap.end(); fIt++)
            fIt->first->setVal(fIt->second->getVal(&x));
      }
      fCache = fPdf->generate(xPrime, fCacheSize);
   }

   bool moved = false;
   if (fMap.size() > 0) {
      moved = !Equals(fLastX, x);

      // if we've moved, set the values of the variables in the PDF to the
      // corresponding values of the variables in x, according to the
      // mappings (i.e. let the variables in x set the given values for the
      // PDF that will generate xPrime)
      if (moved) {
         // update the pdf parameters
         RooStats::SetParameters(&x, &fMaster);

         for (fIt = fMap.begin(); fIt != fMap.end(); fIt++)
            fIt->first->setVal(fIt->second->getVal(&x));

         // save the new x in fLastX
         RooStats::SetParameters(&x, &fLastX);
      }
   }

   // generate new cache if necessary
   if (moved || fCachePosition >= fCacheSize) {
      delete fCache;
      fCache = fPdf->generate(xPrime, fCacheSize);
      fCachePosition = 0;
   }

   const RooArgSet* proposal = fCache->get(fCachePosition);
   fCachePosition++;
   RooStats::SetParameters(proposal, &xPrime);
}

////////////////////////////////////////////////////////////////////////////////
/// Determine whether or not the proposal density is symmetric for
/// points x1 and x2 - that is, whether the probabilty of reaching x2
/// from x1 is equal to the probability of reaching x1 from x2

bool PdfProposal::IsSymmetric(RooArgSet& /* x1 */, RooArgSet& /* x2 */)
{
   // kbelasco: is there a better way to do this?
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the probability of proposing the point x1 given the starting
/// point x2

double PdfProposal::GetProposalDensity(RooArgSet& x1, RooArgSet& x2)
{
   RooStats::SetParameters(&x2, &fMaster);
   for (fIt = fMap.begin(); fIt != fMap.end(); fIt++)
      fIt->first->setVal(fIt->second->getVal(&x2));
   RooArgSet* temp = fPdf->getObservables(x1);
   RooStats::SetParameters(&x1, temp);
   delete temp;
   return fPdf->getVal(&x1); // could just as well use x2
}

////////////////////////////////////////////////////////////////////////////////
/// specify a mapping between a parameter of the proposal function and
/// a parameter of interest.  this mapping is used to set the value of
/// proposalParam equal to the value of update to determine the
/// proposal function.
/// proposalParam is a parameter of the proposal function that must
/// be set to the value of update (from the current point) in order to
/// propose a new point.

void PdfProposal::AddMapping(RooRealVar& proposalParam, RooAbsReal& update)
{
   fMaster.add(*update.getParameters(static_cast<RooAbsData const*>(nullptr)));
   if (update.getParameters(static_cast<RooAbsData const*>(nullptr))->empty())
      fMaster.add(update);
   fMap.insert(pair<RooRealVar*, RooAbsReal*>(&proposalParam, &update));
}
