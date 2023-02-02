// @(#)root/hist:$Id$
// Author: Rene Brun   17/05/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProfile3D
#define ROOT_TProfile3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProfile3D                                                           //
//                                                                      //
// Profile3D histogram class.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH3.h"
#include "TProfile.h"

class TProfile3D : public TH3D {

public:
   friend class TProfileHelper;
   friend class TH1Merger;

protected:
   TArrayD       fBinEntries;      ///< Number of entries per bin
   EErrorType    fErrorMode;       ///< Option to compute errors
   Double_t      fTmin;            ///< Lower limit in T (if set)
   Double_t      fTmax;            ///< Upper limit in T (if set)
   Bool_t        fScaling;         ///<! True when TProfile3D::Scale is called
   Double_t      fTsumwt;          ///< Total Sum of weight*T
   Double_t      fTsumwt2;         ///< Total Sum of weight*T*T
   TArrayD       fBinSumw2;        ///< Array of sum of squares of weights per bin
   static Bool_t fgApproximate;    ///< Bin error approximation option

   Int_t    BufferFill(Double_t, Double_t) override {return -2;} //may not use
   Int_t    BufferFill(Double_t, Double_t, Double_t) override {return -2;} //may not use
   Int_t    BufferFill(Double_t, Double_t, Double_t, Double_t) override {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t x, Double_t y, Double_t z, Double_t t, Double_t w);

   // helper methods for the Merge unification in TProfileHelper
   void SetBins(const Int_t* nbins,const Double_t* range) { SetBins(nbins[0], range[0], range[1],
                                                                    nbins[1], range[2], range[3],
                                                                    nbins[2], range[4], range[5]); };
   Int_t Fill(const Double_t* v) { return Fill(v[0], v[1], v[2], v[3], v[4]); };


   using TH3::Fill;
   Int_t             Fill(Double_t, Double_t,Double_t) override {return TH3::Fill(0); } //MayNotUse
   Int_t             Fill(const char *, const char *, const char *, Double_t) override {return TH3::Fill(0); } //MayNotUse
   Int_t             Fill(const char *, Double_t , const char *, Double_t) override {return TH3::Fill(0); } //MayNotUse
   Int_t             Fill(const char *, const char *, Double_t, Double_t) override {return TH3::Fill(0); } //MayNotUse
   Int_t             Fill(Double_t, const char *, const char *, Double_t) override {return TH3::Fill(0); } //MayNotUse
   Int_t             Fill(Double_t, const char *, Double_t, Double_t) override {return TH3::Fill(0); } //MayNotUse
   Int_t             Fill(Double_t, Double_t, const char *, Double_t) override {return TH3::Fill(0); } //MayNotUse

   Double_t RetrieveBinContent(Int_t bin) const override { return (fBinEntries.fArray[bin] > 0) ? fArray[bin]/fBinEntries.fArray[bin] : 0; }
   //virtual void     UpdateBinContent(Int_t bin, Double_t content);
   Double_t GetBinErrorSqUnchecked(Int_t bin) const override { Double_t err = GetBinError(bin); return err*err; }

   TProfile2D *DoProjectProfile2D(const char* name, const char * title, const TAxis* projX, const TAxis* projY,
                                          bool originalRange, bool useUF, bool useOF) const override;

private:
   Double_t *GetB()  {return &fBinEntries.fArray[0];}
   Double_t *GetB2() {return fBinSumw2.fN ? &fBinSumw2.fArray[0] : nullptr;}
   Double_t *GetW()  {return &fArray[0];}
   Double_t *GetW2() {return &fSumw2.fArray[0];}
   void  SetBins(Int_t, Double_t, Double_t) override
      { MayNotUse("SetBins(Int_t, Double_t, Double_t"); }
   void  SetBins(Int_t, const Double_t*) override
      { MayNotUse("SetBins(Int_t, const Double_t*"); }
   void SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t) override
      { MayNotUse("SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t"); }
   void SetBins(Int_t, const Double_t*, Int_t, const Double_t*) override
      { MayNotUse("SetBins(Int_t, const Double_t*, Int_t, const Double_t*"); }

public:
   TProfile3D();
   TProfile3D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                                ,Int_t nbinsy,Double_t ylow,Double_t yup
                                                ,Int_t nbinsz,Double_t zlow,Double_t zup, Option_t *option="");
   TProfile3D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                                ,Int_t nbinsy,const Double_t *ybins
                                                ,Int_t nbinsz,const Double_t *zbins,Option_t *option="");
   TProfile3D(const TProfile3D &profile);
   TProfile3D &operator=(const TProfile3D &profile);
   ~TProfile3D() override;
   Bool_t    Add(TF1 *h1, Double_t c1=1, Option_t *option="") override;
   Bool_t    Add(const TH1 *h1, Double_t c1=1) override;
   Bool_t    Add(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1) override;
   static  void      Approximate(Bool_t approx=kTRUE);
   void              BuildOptions(Double_t tmin, Double_t tmax, Option_t *option);
   Int_t     BufferEmpty(Int_t action=0) override;
   void      Copy(TObject &hnew) const override;
   Bool_t    Divide(TF1 *h1, Double_t c1=1) override;
   Bool_t    Divide(const TH1 *h1) override;
   Bool_t    Divide(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option="") override;
   void      ExtendAxis(Double_t x, TAxis *axis) override;
   Int_t     Fill(Double_t x, Double_t y, Double_t z, Double_t t) override;
   virtual Int_t     Fill(Double_t x, Double_t y, Double_t z, Double_t t, Double_t w);
   Double_t  GetBinContent(Int_t bin) const override;
   Double_t  GetBinContent(Int_t,Int_t) const override
                     { MayNotUse("GetBinContent(Int_t, Int_t"); return -1; }
   Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const override {return GetBinContent(GetBin(binx,biny,binz));}
   Double_t  GetBinError(Int_t bin) const override;
   Double_t  GetBinError(Int_t,Int_t) const override
                     { MayNotUse("GetBinError(Int_t, Int_t"); return -1; }
   Double_t  GetBinError(Int_t binx, Int_t biny, Int_t binz) const override {return GetBinError(GetBin(binx,biny,binz));}
   virtual Double_t  GetBinEntries(Int_t bin) const;
   virtual Double_t  GetBinEffectiveEntries(Int_t bin);
   virtual TArrayD *GetBinSumw2() {return &fBinSumw2;}
   virtual const TArrayD *GetBinSumw2() const {return &fBinSumw2;}
   Option_t         *GetErrorOption() const;
   void      GetStats(Double_t *stats) const override;
   virtual Double_t  GetTmin() const {return fTmin;}
   virtual Double_t  GetTmax() const {return fTmax;}
   void      LabelsDeflate(Option_t *axis="X") override;
   void      LabelsInflate(Option_t *axis="X") override;
   void      LabelsOption(Option_t *option="h", Option_t *axis="X") override;
   Long64_t  Merge(TCollection *list) override;
   Bool_t    Multiply(TF1 *h1, Double_t c1=1) override;
   Bool_t    Multiply(const TH1 *h1) override;
   Bool_t    Multiply(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option="") override;
   virtual TH3D     *ProjectionXYZ(const char *name="_pxyz", Option_t *option="e") const;
   TProfile2D  *Project3DProfile(Option_t *option="xy") const override; // *MENU*
   void      PutStats(Double_t *stats) override;
   void      Reset(Option_t *option="") override;
   void      SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void      Scale(Double_t c1=1, Option_t *option="") override;
   virtual void      SetBinEntries(Int_t bin, Double_t w);
   void      SetBins(Int_t nbinsx, Double_t xmin, Double_t xmax,
                             Int_t nbinsy, Double_t ymin, Double_t ymax,
                             Int_t nbinsz, Double_t zmin, Double_t zmax) override;
   void      SetBins(Int_t nx, const Double_t *xBins, Int_t ny, const Double_t * yBins, Int_t nz,
                             const Double_t *zBins) override;
   void      SetBinsLength(Int_t n=-1) override;
   void      SetBuffer(Int_t buffersize, Option_t *opt="") override;
   virtual void      SetErrorOption(Option_t *option=""); // *MENU*
   void      Sumw2(Bool_t flag = kTRUE) override;

   ClassDefOverride(TProfile3D,8)  //Profile3D histogram class
};

#endif
