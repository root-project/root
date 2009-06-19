#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "RooStats/UniformProposal.h"
#include "RooAbsCollection.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"

ClassImp(RooStats::UniformProposal);

using namespace RooFit;
using namespace RooStats;

// Populate xPrime with a new proposed point
void UniformProposal::Propose(RooAbsCollection& xPrime)
{
   // kbelasco: remember xPrime has not been checked for containing
   // only RooRealVars
   randomizeCollection(xPrime);
}

// Determine whether or not the proposal density is symmetric for
// points x1 and x2 - that is, whether the probabilty of reaching x2
// from x1 is equal to the probability of reaching x1 from x2
Bool_t UniformProposal::IsSymmetric(RooAbsCollection& /* x1 */ ,
                                    RooAbsCollection& /* x2 */)
{
   return true;
}

// Return the probability of proposing the point xPrime given the starting
// point x
Double_t UniformProposal::GetProposalDensity(RooAbsCollection& /* xPrime */,
                                              RooAbsCollection& x)
{
   // For a uniform proposal, all points have equal probability and the
   // value of the proposal density function is:
   // 1 / (N-dimensional volume of interval)
   Double_t volume = 1.0;
   TIterator* it = x.createIterator();
   RooRealVar* var;
   while ((var = (RooRealVar*)it->Next()) != NULL)
      volume *= (var->getMax() - var->getMin());
   return 1.0 / volume;
}
