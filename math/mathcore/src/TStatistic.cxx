// @(#)root/base:$Id$
// Author: G. Ganis 2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStatistic                                                           //
//                                                                      //
// Statistical variable, defined by its mean and RMS.                   //
// Named, streamable, storable and mergeable.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStatistic.h"


templateClassImp(TStatistic)

//______________________________________________________________________________
TStatistic::TStatistic(const char *name, Int_t n, const Double_t *val, const Double_t *w)
         : fName(name), fN(0), fW(0.), fW2(0.), fM(0.), fM2(0.)
{
   // Constructor from a vector of values

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
   // Merge objects in the list.
   // Returns the number of objects that were in the list.
   TIter nxo(in);
   Int_t n = 0;
   while (TObject *o = nxo()) {
      TStatistic *c = dynamic_cast<TStatistic *>(o);
      if (c) {
         if (fW == 0 || c->fW == 0 || ((fW + c->fW) == 0) ) {
            Error("Merge","Zero sum of weights - cannot merge data from %s",c->GetName() );
            continue;
         }
         double temp = (c->fW)/(fW) * fM - c->fM; 
         fM2 += c->fM2 + fW/(c->fW*(c->fW + fW) ) * temp * temp;  
         fM  += c->fM;
         fW  += c->fW;
         fW2 += c->fW2;
         fN  += c->fN;
         n++;
      }
   }
   return n;
}
