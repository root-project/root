// @(#)root/hist:$Id$
// Author: Filip Ilic

#ifndef ROOT_TProfile2Poly
#define ROOT_TProfile2Poly

#include "TH2Poly.h"
#include "TProfile.h"

class TProfile2PolyBin : public TH2PolyBin {
public:
   friend class TProfile2Poly;

   TProfile2PolyBin();
   TProfile2PolyBin(TObject *poly, Int_t bin_number);

   virtual ~TProfile2PolyBin() {}

   void Update();
   void UpdateAverage();
   void UpdateError();
   void ClearStats();

private:
   void Fill(Double_t value, Double_t weight);

   Double_t fSumv;
   Double_t fSumw;
   Double_t fSumvw;
   Double_t fSumv2;
   Double_t fSumwv2;

   Double_t fAverage;
   Double_t fError;

   ClassDef(TProfile2PolyBin, 1)
};

class TProfile2Poly : public TH2Poly {

public:
   friend class TProfileHelper;

   TProfile2Poly() {}
   TProfile2Poly(const char *name, const char *title, Double_t xlow, Double_t xup, Double_t ylow, Double_t yup);

   TProfile2Poly(const char *name, const char *title, Int_t nX, Double_t xlow, Double_t xup, Int_t nY, Double_t ylow,
                 Double_t yup);

   virtual ~TProfile2Poly() {}

   virtual TProfile2PolyBin *CreateBin(TObject *poly) override;

   using TH2Poly::Fill;
   virtual Int_t Fill(Double_t xcoord, Double_t ycoord, Double_t value) override;
   virtual Int_t Fill(Double_t xcoord, Double_t ycoord, Double_t value, Double_t weight);

   Long64_t Merge(std::vector<TProfile2Poly *> list);
   Long64_t Merge(TCollection *in) override;
   virtual void Reset(Option_t *option = "") override;

   Int_t GetOverflowRegionFromCoordinates(Double_t x, Double_t y);
   Int_t OverflowIdxToArrayIdx(Int_t val) { return -val - 1; }

   // option to dispay different measures on bins
   void SetContentToAverage(); // this one is used by default
   void SetContentToError();

   Double_t GetOverflowContent(Int_t idx) { return regions[idx].fSumw; }
   void printOverflowRegions();

private:
   Double_t fTsumwz;
   Double_t fTsumwz2;

   TProfile2PolyBin regions[kNOverflow];

   ClassDefOverride(TProfile2Poly, 1)
};
#endif
