// @(#)root/hist:$Id$
// Author: Filip Ilic

#ifndef ROOT_TProfile2Poly
#define ROOT_TProfile2Poly

#include "TH2Poly.h"
#include "TProfile.h"

class TProfile2PolyBin : public TH2PolyBin {
public:
   TProfile2PolyBin();
   TProfile2PolyBin(TObject *poly, Int_t bin_number);

   virtual ~TProfile2PolyBin() {}

   void Update();
   void UpdateAverage();
   void UpdateError();

   Double_t GetError() { return fError; }
   Double_t GetAverage() { return fAverage; }

   Double_t GetFSumw() const { return fSumw; }
   Double_t GetFSumw2() const { return fSumw2; }
   Double_t GetFSumwz() const { return fSumwz; }
   Double_t GetFSumwz2() const { return fSumwz2; }
   Double_t GetFNumEntries() const { return fNumEntries; }

   void SetFSumw(Double_t value) { fSumw = value; }
   void SetFSumw2(Double_t value) { fSumw2 = value; }
   void SetFSumwz(Double_t value) { fSumwz = value; }
   void SetFSumwz2(Double_t value) { fSumwz2 = value; }
   void SetFNumEntries(Double_t value) { fNumEntries = value; }

   void ClearStats();

private:
   Double_t fSumw;
   Double_t fSumw2;
   Double_t fSumwz;
   Double_t fSumwz2;
   Double_t fNumEntries;

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
   Int_t OverflowIdxToArrayIdx(Int_t val) { return -val-1; }

   // option to dispay different measures on bins
   void SetContentToAverageW(); // this one is used by default
   void SetContentToErrorW();

private:
   Double_t fTsumwz;
   Double_t fTsumwz2;

   ClassDefOverride(TProfile2Poly, 1)
};
#endif
