// @(#)root/hist:$Id$
// Author: Filip Ilic

#ifndef ROOT_TProfile2Poly
#define ROOT_TProfile2Poly

#include "TH2Poly.h"

class TProfile2PolyBin: public TH2PolyBin {
public:
    TProfile2PolyBin();
    TProfile2PolyBin(TObject* poly, Int_t bin_number);

    virtual ~TProfile2PolyBin();

    void UpdateAverage();

    Double_t getFSumV() const       { return fSumV;   }
    Double_t getFSumV2() const      { return fSumV2;  }
    Double_t getFSumVW() const      { return fSumVW;  }
    Double_t getFSumVW2() const     { return fSumVW2; }
    Double_t getFNumEntries() const { return fSumVW2; }

public:
    Double_t fSumV;
    Double_t fSumV2;
    Double_t fSumVW;
    Double_t fSumVW2;
    Double_t fNumEntries;

    ClassDef(TProfile2PolyBin,1)
};


class TProfile2Poly: public TH2Poly {
public:
    TProfile2Poly(){}
    TProfile2Poly(const char *name, const char *title,
                  Double_t xlow, Double_t xup,
                  Double_t ylow, Double_t yup);

    TProfile2Poly(const char *name,const char *title,
                  Int_t nX, Double_t xlow, Double_t xup,
                  Int_t nY, Double_t ylow, Double_t yup);

    virtual ~TProfile2Poly();

    using TH2Poly::AddBin;
    virtual Int_t AddBin(TObject *poly) override;

    using TH2Poly::Fill;
    virtual Int_t Fill(Double_t xcoord, Double_t ycoord, Double_t value) override;
    virtual Int_t Fill(Double_t xcoord, Double_t ycoord, Double_t value, Double_t weight);

    //virtual Long64_t  Merge(TCollection *list);

    ClassDefOverride(TProfile2Poly,1)
};
#endif
