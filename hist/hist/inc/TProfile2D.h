// @(#)root/hist:$Id$
// Author: Rene Brun   16/04/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProfile2D
#define ROOT_TProfile2D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProfile2D                                                           //
//                                                                      //
// Profile2D histogram class.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH2.h"
#include "TProfile.h"

class TProfile2D : public TH2D {

public:
   friend class TProfileHelper;
   friend class TH1Merger;

protected:
   TArrayD     fBinEntries;       ///< Number of entries per bin
   EErrorType  fErrorMode;        ///< Option to compute errors
   Double_t    fZmin;             ///< Lower limit in Z (if set)
   Double_t    fZmax;             ///< Upper limit in Z (if set)
   Bool_t      fScaling;          ///<! True when TProfile2D::Scale is called
   Double_t    fTsumwz;           ///< Total Sum of weight*Z
   Double_t    fTsumwz2;          ///< Total Sum of weight*Z*Z
   TArrayD     fBinSumw2;         ///< Array of sum of squares of weights per bin
   static Bool_t   fgApproximate; ///< Bin error approximation option

   Int_t    BufferFill(Double_t, Double_t) override {return -2;} //may not use
   Int_t    BufferFill(Double_t, Double_t, Double_t) override {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t x, Double_t y, Double_t z, Double_t w);

   // helper methods for the Merge unification in TProfileHelper
   void SetBins(const Int_t* nbins, const Double_t* range) { SetBins(nbins[0], range[0], range[1],
                                                                     nbins[1], range[2], range[3]); };
   Int_t Fill(const Double_t* v) { return Fill(v[0], v[1], v[2], v[3]); };

   TProfile *DoProfile(bool onX, const char *name, Int_t firstbin, Int_t lastbin, Option_t *option) const override;

   using TH2::Fill;
   Int_t             Fill(Double_t, Double_t) override {return TH2::Fill(0); } //MayNotUse

   Double_t RetrieveBinContent(Int_t bin) const override { return (fBinEntries.fArray[bin] > 0) ? fArray[bin]/fBinEntries.fArray[bin] : 0; }
   //virtual void     UpdateBinContent(Int_t bin, Double_t content);
   Double_t GetBinErrorSqUnchecked(Int_t bin) const override { Double_t err = GetBinError(bin); return err*err; }

private:
   Double_t *GetB()  {return &fBinEntries.fArray[0];}
   Double_t *GetB2() {return fBinSumw2.fN ? &fBinSumw2.fArray[0] : nullptr; }
   Double_t *GetW()  {return &fArray[0];}
   Double_t *GetW2() {return &fSumw2.fArray[0];}
   void  SetBins(Int_t, Double_t, Double_t) override
      { MayNotUse("SetBins(Int_t, Double_t, Double_t"); }
   void  SetBins(Int_t, const Double_t*) override
      { MayNotUse("SetBins(Int_t, const Double_t*"); }
   void SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t) override
      { MayNotUse("SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t"); }
   void SetBins(Int_t, const Double_t *, Int_t, const Double_t *, Int_t, const Double_t *) override
      { MayNotUse("SetBins(Int_t, const Double_t*, Int_t, const Double_t*, Int_t, const Double_t*"); }

public:
   TProfile2D();
   TProfile2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                               ,Int_t nbinsy,Double_t ylow,Double_t yup
                               ,Double_t zlow, Double_t zup,Option_t *option="");
   TProfile2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                               ,Int_t nbinsy,Double_t ylow,Double_t yup,Option_t *option="");
   TProfile2D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                               ,Int_t nbinsy,Double_t ylow,Double_t yup,Option_t *option="");
   TProfile2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                               ,Int_t nbinsy,const Double_t *ybins,Option_t *option="");
   TProfile2D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                               ,Int_t nbinsy,const Double_t *ybins,Option_t *option="");
   TProfile2D(const TProfile2D &profile);
   TProfile2D &operator=(const TProfile2D &profile);
   ~TProfile2D() override;
   Bool_t    Add(TF1 *h1, Double_t c1=1, Option_t *option="") override;
   Bool_t    Add(const TH1 *h1, Double_t c1=1) override;
   Bool_t    Add(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1) override; // *MENU*
   static  void      Approximate(Bool_t approx=kTRUE);
   void              BuildOptions(Double_t zmin, Double_t zmax, Option_t *option);
   Int_t     BufferEmpty(Int_t action=0) override;
   void      Copy(TObject &hnew) const override;
   Bool_t    Divide(TF1 *h1, Double_t c1=1) override;
   Bool_t    Divide(const TH1 *h1) override;
   Bool_t    Divide(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option="") override; // *MENU*
   void      ExtendAxis(Double_t x, TAxis *axis) override;
   Int_t             Fill(Double_t x, Double_t y, Double_t z) override;
   virtual Int_t     Fill(Double_t x, const char *namey, Double_t z, Double_t w = 1.);
   virtual Int_t     Fill(const char *namex, Double_t y, Double_t z, Double_t w = 1.);
   virtual Int_t     Fill(const char *namex, const char *namey, Double_t z, Double_t w = 1.);
   virtual Int_t     Fill(Double_t x, Double_t y, Double_t z, Double_t w);
   Double_t  GetBinContent(Int_t bin) const override;
   Double_t  GetBinContent(Int_t binx, Int_t biny) const override {return GetBinContent(GetBin(binx,biny));}
   Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const override {return GetBinContent(GetBin(binx,biny));}
   Double_t  GetBinError(Int_t bin) const override;
   Double_t  GetBinError(Int_t binx, Int_t biny) const override {return GetBinError(GetBin(binx,biny));}
   Double_t  GetBinError(Int_t binx, Int_t biny, Int_t) const override {return GetBinError(GetBin(binx,biny));}
   virtual Double_t  GetBinEntries(Int_t bin) const;
   virtual Double_t  GetBinEffectiveEntries(Int_t bin);
   virtual TArrayD *GetBinSumw2() {return &fBinSumw2;}
   virtual const TArrayD *GetBinSumw2() const {return &fBinSumw2;}
   Option_t         *GetErrorOption() const;
   void      GetStats(Double_t *stats) const override;
   virtual Double_t  GetZmin() const {return fZmin;}
   virtual Double_t  GetZmax() const {return fZmax;}
   void      LabelsDeflate(Option_t *axis="X") override;
   void      LabelsInflate(Option_t *axis="X") override;
   void      LabelsOption(Option_t *option="h", Option_t *axis="X") override;
   Long64_t  Merge(TCollection *list) override;
   Bool_t    Multiply(TF1 *h1, Double_t c1=1) override;
   Bool_t    Multiply(const TH1 *h1) override;
   Bool_t    Multiply(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option="") override; // *MENU*
   TH2D             *ProjectionXY(const char *name="_pxy", Option_t *option="e") const;
   TProfile         *ProfileX(const char *name="_pfx", Int_t firstybin=0, Int_t lastybin=-1, Option_t *option="") const;   // *MENU*
   TProfile         *ProfileY(const char *name="_pfy", Int_t firstxbin=0, Int_t lastxbin=-1, Option_t *option="") const;   // *MENU*
   void      PutStats(Double_t *stats) override;
   void      Reset(Option_t *option="") override;
   TProfile2D *Rebin2D(Int_t nxgroup=2, Int_t nygroup=2, const char *newname="") override;
   TProfile2D *RebinX(Int_t ngroup=2, const char *newname="") override;
   TProfile2D *RebinY(Int_t ngroup=2, const char *newname="") override;
   void      SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void      Scale(Double_t c1=1, Option_t *option="") override;
   virtual void      SetBinEntries(Int_t bin, Double_t w);
   void      SetBins(Int_t nbinsx, Double_t xmin, Double_t xmax, Int_t nbinsy, Double_t ymin, Double_t ymax) override;
   void      SetBins(Int_t nx, const Double_t *xBins, Int_t ny, const Double_t *yBins) override;
   void      SetBinsLength(Int_t n=-1) override;
   void      SetBuffer(Int_t buffersize, Option_t *option="") override;
   virtual void      SetErrorOption(Option_t *option=""); // *MENU*
   void      Sumw2(Bool_t flag = kTRUE) override;
   Double_t GetNumberOfBins() { return fBinEntries.GetSize(); }

   ClassDefOverride(TProfile2D,8)  //Profile2D histogram class
};

#endif
