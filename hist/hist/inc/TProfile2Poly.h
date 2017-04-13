// @(#)root/hist:$Id$
// Author: Filip Ilic

#ifndef ROOT_TProfile2Poly
#define ROOT_TProfile2Poly

#include "TH2Poly.h"
#include "TProfile.h"

class TProfile2PolyBin: public TH2PolyBin {
public:
    TProfile2PolyBin();
    TProfile2PolyBin(TObject* poly, Int_t bin_number);

    virtual ~TProfile2PolyBin();

    void UpdateAverage();

    Double_t getFSumV()       const { return fSumV;       }
    Double_t getFSumV2()      const { return fSumV2;      }
    Double_t getFSumVW()      const { return fSumVW;      }
    Double_t getFSumVW2()     const { return fSumVW2;     }
    Double_t getFNumEntries() const { return fNumEntries; }

    void setFSumV(Double_t value)       { fSumV = value;       }
    void setFSumV2(Double_t value)      { fSumV2 = value;      }
    void setFSumVW(Double_t value)      { fSumVW = value;      }
    void setFSumVW2(Double_t value)     { fSumVW2 = value;     }
    void setFNumEntries(Double_t value) { fNumEntries = value; }

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
    friend class TProfileHelper;

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

    void Merge(std::vector<TProfile2Poly*> list);

    ClassDefOverride(TProfile2Poly,1)
};
#endif
