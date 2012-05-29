// @(#)root/mathcore:$Id$
// Author: G. Ganis 2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStatistic
#define ROOT_TStatistic


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStatistic                                                           //
//                                                                      //
// Statistical variable, defined by its mean, RMS and related errors.   //
// Named, streamable, storable and mergeable.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TCollection
#include "TCollection.h"
#endif

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

class TStatistic : public TObject {

private:
   TString     fName;
   Long64_t    fN;       // Number of fills
   Double_t    fW;       // Sum of weights    
   Double_t    fW2;      // Sum of weights**2    
   Double_t    fMean;    // Mean
   Double_t    fM2;      // Second order momentum

public:
   TStatistic(const char *name = "") : fName(name), fN(0), fW(0.), fW2(0.), fMean(0.), fM2(0.) { }
   TStatistic(const char *name, Int_t n, const Double_t *val, const Double_t *w = 0);
   ~TStatistic() { }
 
   // Getters
   const char    *GetName() const { return fName; }
   ULong_t        Hash() const { return fName.Hash(); }

   inline const Long64_t &GetN() const { return fN; }
   inline const Double_t &GetM2() const { return fM2; }
   inline const Double_t &GetMean() const { return fMean; }
   inline       Double_t  GetMeanErr() const { if (fW > 0.) return TMath::Sqrt(fM2 / fW2 / fW);
                                               return 0.; }
   inline       Double_t  GetRMS() const { if (fW > 0.) { return TMath::Sqrt(fM2 / fW); } return -1; }
   inline       Double_t  GetVar() const { if (fW > 0.) {
                                            if (fN > 1) { return (fM2 / fW)*(fN / (fN-1.)); }
                                            return 0; } return -1.; }
   inline       Double_t  GetVarN() const { if (fW > 0.) { return fM2 / fW; } return -1.; }
   inline const Double_t &GetW() const { return fW; }
   inline const Double_t &GetW2() const { return fW2; }

   // Merging
   Int_t Merge(TCollection *in);

   // Fill
   inline void Fill(const Double_t &val, Double_t w = 1.) {
      fN++;
      // Incremental quantities
      Double_t tW = w + fW;
      Double_t dt = val - fMean ;
      Double_t rr = dt * w / tW ;
      fMean += rr;
      fM2 += fW * dt * rr;
      fW = tW;
      fW2 += w*w;
   }

   // Print
   void ls(Option_t *opt = "") const { Print(opt); }
   void Print(Option_t * = "") const {
      // Print this parameter content
      TROOT::IndentLevel();
      Printf(" OBJ: TStatistic\t %s = %.3g +- %.3g \t RMS = %.3g \t N = %lld",
             fName.Data(), fMean, GetMeanErr(), GetRMS(), fN);
   }
   
   ClassDef(TStatistic,1)  //Named statistical variable
};

// Implementation of Merge
inline Int_t TStatistic::Merge(TCollection *in) {
   // Merge objects in the list.
   // Returns the number of objects that were in the list.
   TIter nxo(in);
   Int_t n = 0;
   while (TObject *o = nxo()) {
      TStatistic *c = dynamic_cast<TStatistic *>(o);
      if (c) {
         fMean += c->GetMean();
         fW += c->GetW();
         fW2 += c->GetW2();
         fM2 += c->GetM2();
         fN += c->GetN();
         n++;
      }
   }
   return n;
}
#endif
