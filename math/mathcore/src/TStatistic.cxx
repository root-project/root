// @(#)root/base:$Id$
// Author: G. Ganis 2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/**
                                                               
\class TStatistic                                                     
                                                               
Statistical variable, defined by its mean and variance (RMS).             
Named, streamable, storable and mergeable.                     

@ingroup MathCore

*/
                                                               


#include "TStatistic.h"


templateClassImp(TStatistic);

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a vector of values

TStatistic::TStatistic(const char *name, Int_t n, const Double_t *val, const Double_t *w)
         : fName(name), fN(0), fW(0.), fW2(0.), fM(0.), fM2(0.)
{
   if (n > 0) {
      for (Int_t i = 0; i < n; i++) {
         if (w) {
            Fill(val[i], w[i]);
         } else {
            Fill(val[i]);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// TStatistic destructor.

TStatistic::~TStatistic()
{
   // Required since we overload TObject::Hash.
   ROOT::CallRecursiveRemoveIfNeeded(*this);
}

void TStatistic::Fill(Double_t val, Double_t w) {
      // Incremental quantities
   // use formula 1.4 in Chan-Golub, LeVeque
   // Algorithms for computing the Sample Variance (1983)
   // genralized by LM for the case of weights 

   if (w == 0) return;

   fN++;

   Double_t tW = fW + w;
   fM += w * val; 

//      Double_t dt = val - fM ;
   if (tW == 0) {
      Warning("Fill","Sum of weights is zero - ignore current data point");
      fN--;
      return;
   }
   if (fW != 0) {  // from the second time
      Double_t rr = ( tW * val - fM);
      fM2 += w * rr * rr / (tW * fW);
   }
   fW = tW;
   fW2 += w*w;
}


void TStatistic::Print(Option_t *) const {
   // Print this parameter content
   TROOT::IndentLevel();
   Printf(" OBJ: TStatistic\t %s = %.5g +- %.4g \t RMS = %.5g \t N = %lld",
          fName.Data(), GetMean(), GetMeanErr(), GetRMS(), fN);
}


// Implementation of Merge
Int_t TStatistic::Merge(TCollection *in) {

   // Let's organise the list of objects to merge excluding the empty ones
   std::vector<TStatistic*> statPtrs;
   if (this->fN != 0LL) statPtrs.push_back(this);
   TStatistic *statPtr;
   for (auto o : *in) {
      if ((statPtr = dynamic_cast<TStatistic *>(o)) && statPtr->fN != 0LL) {
         statPtrs.push_back(statPtr);
      }
   }

   // No object included this has entries
   const auto nStatsPtrs = statPtrs.size();

   // Early return possible in case nothing has been filled
   if (nStatsPtrs == 0) return 0;

   // Merge the statistic quantities into local variables to then 
   // update the data members of this object
   auto firstStatPtr = statPtrs[0];
   auto N = firstStatPtr->fN;
   auto M = firstStatPtr->fM;
   auto M2 = firstStatPtr->fM2;
   auto W = firstStatPtr->fW;
   auto W2 = firstStatPtr->fW2;
   for (auto i = 1U; i < nStatsPtrs; ++i) {
      auto c = statPtrs[i];
      double temp = (c->fW) / (W)*M - c->fM;
      M2 += c->fM2 + W / (c->fW * (c->fW + W)) * temp * temp;
      M += c->fM;
      W += c->fW;
      W2 += c->fW2;
      N += c->fN;
   }

   // Now update the data members of this object
   fN = N;
   fW = W;
   fW2 = W2;
   fM = M;
   fM2 = M2;

   return nStatsPtrs;

}
