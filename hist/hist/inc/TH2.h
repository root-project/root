// @(#)root/hist:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH2
#define ROOT_TH2


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TH2                                                                  //
//                                                                      //
// 2-Dim histogram base class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH1.h"
#include "TMatrixFBasefwd.h"
#include "TMatrixDBasefwd.h"

class TProfile;

class TH2 : public TH1 {

protected:
   Double_t     fScalefactor;     ///< Scale factor
   Double_t     fTsumwy;          ///< Total Sum of weight*Y
   Double_t     fTsumwy2;         ///< Total Sum of weight*Y*Y
   Double_t     fTsumwxy;         ///< Total Sum of weight*X*Y

   TH2();
   TH2(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                         ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                         ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                         ,Int_t nbinsy,const Double_t *ybins);
   TH2(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                         ,Int_t nbinsy,const Double_t *ybins);
   TH2(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins
                                         ,Int_t nbinsy,const Float_t  *ybins);

   virtual Int_t     BufferFill(Double_t x, Double_t y, Double_t w);
   virtual TH1D     *DoProjection(bool onX, const char *name, Int_t firstbin, Int_t lastbin, Option_t *option) const;
   virtual TProfile *DoProfile(bool onX, const char *name, Int_t firstbin, Int_t lastbin, Option_t *option) const;
   virtual TH1D     *DoQuantiles(bool onX, const char *name, Double_t prob) const;
   virtual void      DoFitSlices(bool onX, TF1 *f1, Int_t firstbin, Int_t lastbin, Int_t cut, Option_t *option, TObjArray* arr);

   Int_t    BufferFill(Double_t, Double_t) override {return -2;} //may not use
   Int_t    Fill(Double_t) override; //MayNotUse
   Int_t    Fill(const char*, Double_t) override { return Fill(0);}  //MayNotUse

   Double_t Interpolate(Double_t x) const override; // may not use

private:

   TH2(const TH2&) = delete;
   TH2& operator=(const TH2&) = delete;

   // make private methods which have a TH1 signature and should not
   using TH1::Integral;
   using TH1::IntegralAndError;

public:
   virtual ~TH2();
           Int_t    BufferEmpty(Int_t action=0) override;
           void     Copy(TObject &hnew) const override;
           Int_t    Fill(Double_t x, Double_t y) override;
   virtual Int_t    Fill(Double_t x, Double_t y, Double_t w);
   virtual Int_t    Fill(Double_t x, const char *namey, Double_t w);
   virtual Int_t    Fill(const char *namex, Double_t y, Double_t w);
   virtual Int_t    Fill(const char *namex, const char *namey, Double_t w);
           void     FillN(Int_t, const Double_t *, const Double_t *, Int_t) override {} //MayNotUse
           void     FillN(Int_t ntimes, const Double_t *x, const Double_t *y, const Double_t *w, Int_t stride=1) override;
           void     FillRandom(const char *fname, Int_t ntimes=5000, TRandom *rng = nullptr) override;
           void     FillRandom(TH1 *h, Int_t ntimes=5000, TRandom *rng = nullptr) override;
   virtual void     FitSlicesX(TF1 *f1 = nullptr, Int_t firstybin=0, Int_t lastybin=-1, Int_t cut=0, Option_t *option="QNR", TObjArray* arr = nullptr);
   virtual void     FitSlicesY(TF1 *f1 = nullptr, Int_t firstxbin=0, Int_t lastxbin=-1, Int_t cut=0, Option_t *option="QNR", TObjArray* arr = nullptr);
           Int_t    GetBin(Int_t binx, Int_t biny, Int_t binz = 0) const override;
   virtual Double_t GetBinWithContent2(Double_t c, Int_t &binx, Int_t &biny, Int_t firstxbin=1, Int_t lastxbin=-1,Int_t firstybin=1, Int_t lastybin=-1, Double_t maxdiff=0) const;
   using TH1::GetBinContent;
           Double_t GetBinContent(Int_t binx, Int_t biny) const override { return TH1::GetBinContent( GetBin(binx, biny) ); }
           Double_t GetBinContent(Int_t binx, Int_t biny, Int_t) const override { return TH1::GetBinContent( GetBin(binx, biny) ); }
   using TH1::GetBinErrorLow;
   using TH1::GetBinErrorUp;
   virtual Double_t GetBinErrorLow(Int_t binx, Int_t biny) { return TH1::GetBinErrorLow( GetBin(binx, biny) ); }
   virtual Double_t GetBinErrorUp(Int_t binx, Int_t biny) { return TH1::GetBinErrorUp( GetBin(binx, biny) ); }
   virtual Double_t GetCorrelationFactor(Int_t axis1=1,Int_t axis2=2) const;
   virtual Double_t GetCovariance(Int_t axis1=1,Int_t axis2=2) const;
   virtual void     GetRandom2(Double_t &x, Double_t &y, TRandom * rng = nullptr);
           void     GetStats(Double_t *stats) const override;
           Double_t Integral(Option_t *option="") const override;
   //virtual Double_t Integral(Int_t, Int_t, Option_t * ="") const {return 0;}
   virtual Double_t Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Option_t *option="") const;
   virtual Double_t Integral(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, Option_t * ="") const {return 0;}
   virtual Double_t IntegralAndError(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Double_t & err, Option_t *option="") const;
           Double_t Interpolate(Double_t x, Double_t y) const override;
           Double_t Interpolate(Double_t x, Double_t y, Double_t z) const override;
           Double_t KolmogorovTest(const TH1 *h2, Option_t *option="") const override;
           TH2     *RebinX(Int_t ngroup=2, const char *newname="") override; // *MENU*
   virtual TH2     *RebinY(Int_t ngroup=2, const char *newname=""); // *MENU*
           TH2     *Rebin(Int_t ngroup=2, const char*newname="", const Double_t *xbins = nullptr) override;  // re-implementation of the TH1 function using RebinX
   virtual TH2     *Rebin2D(Int_t nxgroup=2, Int_t nygroup=2, const char *newname=""); // *MENU*
          TProfile *ProfileX(const char *name="_pfx", Int_t firstybin=1, Int_t lastybin=-1, Option_t *option="") const;   // *MENU*
          TProfile *ProfileY(const char *name="_pfy", Int_t firstxbin=1, Int_t lastxbin=-1, Option_t *option="") const;   // *MENU*
           TH1D    *ProjectionX(const char *name="_px", Int_t firstybin=0, Int_t lastybin=-1, Option_t *option="") const; // *MENU*
           TH1D    *ProjectionY(const char *name="_py", Int_t firstxbin=0, Int_t lastxbin=-1, Option_t *option="") const; // *MENU*
           void     PutStats(Double_t *stats) override;
           TH1D    *QuantilesX(Double_t prob = 0.5, const char * name = "_qx" ) const;
           TH1D    *QuantilesY(Double_t prob = 0.5, const char * name = "_qy" ) const;
           void     Reset(Option_t *option="") override;
           void     SetBinContent(Int_t bin, Double_t content) override;
           void     SetBinContent(Int_t binx, Int_t biny, Double_t content) override { SetBinContent(GetBin(binx, biny), content); }
           void     SetBinContent(Int_t binx, Int_t biny, Int_t, Double_t content) override { SetBinContent(GetBin(binx, biny), content); }
   virtual void     SetShowProjectionX(Int_t nbins=1);  // *MENU*
   virtual void     SetShowProjectionY(Int_t nbins=1);  // *MENU*
           TH1     *ShowBackground(Int_t niter=20, Option_t *option="same") override;
           Int_t    ShowPeaks(Double_t sigma=2, Option_t *option="", Double_t threshold=0.05) override; // *MENU*
           void     Smooth(Int_t ntimes=1, Option_t *option="") override; // *MENU*

   ClassDefOverride(TH2,5)  //2-Dim histogram base class
};


//______________________________________________________________________________

class TH2C : public TH2, public TArrayC {

public:
   TH2C();
   TH2C(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2C(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2C(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins
                                          ,Int_t nbinsy,const Float_t  *ybins);
   TH2C(const TH2C &h2c);
   virtual ~TH2C();

           void     AddBinContent(Int_t bin) override;
           void     AddBinContent(Int_t bin, Double_t w) override;
           void     Copy(TObject &hnew) const override;
           void     Reset(Option_t *option="") override;
           void     SetBinsLength(Int_t n=-1) override;

           TH2C&    operator=(const TH2C &h1);
   friend  TH2C     operator*(Float_t c1, TH2C &h1);
   friend  TH2C     operator*(TH2C &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2C     operator+(TH2C &h1, TH2C &h2);
   friend  TH2C     operator-(TH2C &h1, TH2C &h2);
   friend  TH2C     operator*(TH2C &h1, TH2C &h2);
   friend  TH2C     operator/(TH2C &h1, TH2C &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Char_t (content); }

   ClassDefOverride(TH2C,4)  //2-Dim histograms (one char per channel)
};


//______________________________________________________________________________

class TH2S : public TH2, public TArrayS {

public:
   TH2S();
   TH2S(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2S(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2S(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins
                                          ,Int_t nbinsy,const Float_t  *ybins);
   TH2S(const TH2S &h2s);
   virtual ~TH2S();

           void     AddBinContent(Int_t bin) override;
           void     AddBinContent(Int_t bin, Double_t w) override;
           void     Copy(TObject &hnew) const override;
           void     Reset(Option_t *option="") override;
           void     SetBinsLength(Int_t n=-1) override;

           TH2S&    operator=(const TH2S &h1);
   friend  TH2S     operator*(Float_t c1, TH2S &h1);
   friend  TH2S     operator*(TH2S &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2S     operator+(TH2S &h1, TH2S &h2);
   friend  TH2S     operator-(TH2S &h1, TH2S &h2);
   friend  TH2S     operator*(TH2S &h1, TH2S &h2);
   friend  TH2S     operator/(TH2S &h1, TH2S &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Short_t (content); }

   ClassDefOverride(TH2S,4)  //2-Dim histograms (one short per channel)
};


//______________________________________________________________________________

class TH2I : public TH2, public TArrayI {

public:
   TH2I();
   TH2I(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2I(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2I(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2I(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2I(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins
                                          ,Int_t nbinsy,const Float_t  *ybins);
   TH2I(const TH2I &h2i);
   virtual ~TH2I();

           void     AddBinContent(Int_t bin) override;
           void     AddBinContent(Int_t bin, Double_t w) override;
           void     Copy(TObject &hnew) const override;
           void     Reset(Option_t *option="") override;
           void     SetBinsLength(Int_t n=-1) override;

           TH2I&    operator=(const TH2I &h1);
   friend  TH2I     operator*(Float_t c1, TH2I &h1);
   friend  TH2I     operator*(TH2I &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2I     operator+(TH2I &h1, TH2I &h2);
   friend  TH2I     operator-(TH2I &h1, TH2I &h2);
   friend  TH2I     operator*(TH2I &h1, TH2I &h2);
   friend  TH2I     operator/(TH2I &h1, TH2I &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Int_t (content); }

   ClassDefOverride(TH2I,4)  //2-Dim histograms (one 32 bits integer per channel)
};


//______________________________________________________________________________

class TH2F : public TH2, public TArrayF {

public:
   TH2F();
   TH2F(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2F(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2F(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins
                                          ,Int_t nbinsy,const Float_t  *ybins);
   TH2F(const TMatrixFBase &m);
   TH2F(const TH2F &h2f);
   virtual ~TH2F();

           void     AddBinContent(Int_t bin) override {++fArray[bin];}
           void     AddBinContent(Int_t bin, Double_t w) override
                                 {fArray[bin] += Float_t (w);}
           void     Copy(TObject &hnew) const override;
           void     Reset(Option_t *option="") override;
           void     SetBinsLength(Int_t n=-1) override;

           TH2F&    operator=(const TH2F &h1);
   friend  TH2F     operator*(Float_t c1, TH2F &h1);
   friend  TH2F     operator*(TH2F &h1, Float_t c1);
   friend  TH2F     operator+(TH2F &h1, TH2F &h2);
   friend  TH2F     operator-(TH2F &h1, TH2F &h2);
   friend  TH2F     operator*(TH2F &h1, TH2F &h2);
   friend  TH2F     operator/(TH2F &h1, TH2F &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Float_t (content); }

   ClassDefOverride(TH2F,4)  //2-Dim histograms (one float per channel)
};


//______________________________________________________________________________

class TH2D : public TH2, public TArrayD {

public:
   TH2D();
   TH2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,Double_t ylow,Double_t yup);
   TH2D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins);
   TH2D(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins
                                          ,Int_t nbinsy,const Float_t  *ybins);
   TH2D(const TMatrixDBase &m);
   TH2D(const TH2D &h2d);
   virtual ~TH2D();

           void     AddBinContent(Int_t bin) override {++fArray[bin];}
           void     AddBinContent(Int_t bin, Double_t w) override
                                 {fArray[bin] += Double_t (w);}
           void     Copy(TObject &hnew) const override;
           void     Reset(Option_t *option="") override;
           void     SetBinsLength(Int_t n=-1) override;

           TH2D&    operator=(const TH2D &h1);
   friend  TH2D     operator*(Float_t c1, TH2D &h1);
   friend  TH2D     operator*(TH2D &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2D     operator+(TH2D &h1, TH2D &h2);
   friend  TH2D     operator-(TH2D &h1, TH2D &h2);
   friend  TH2D     operator*(TH2D &h1, TH2D &h2);
   friend  TH2D     operator/(TH2D &h1, TH2D &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return fArray[bin]; }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = content; }

   ClassDefOverride(TH2D,4)  //2-Dim histograms (one double per channel)
};

#endif
