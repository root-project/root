// @(#)root/hist:$Id$
// Author: Rene Brun   27/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH3
#define ROOT_TH3


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TH3                                                                  //
//                                                                      //
// 3-Dim histogram base class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH1.h"

#include "TAtt3D.h"

class TH2D;
class TProfile2D;

class TH3 : public TH1, public TAtt3D {

protected:
   Double_t     fTsumwy;          ///< Total Sum of weight*Y
   Double_t     fTsumwy2;         ///< Total Sum of weight*Y*Y
   Double_t     fTsumwxy;         ///< Total Sum of weight*X*Y
   Double_t     fTsumwz;          ///< Total Sum of weight*Z
   Double_t     fTsumwz2;         ///< Total Sum of weight*Z*Z
   Double_t     fTsumwxz;         ///< Total Sum of weight*X*Z
   Double_t     fTsumwyz;         ///< Total Sum of weight*Y*Z

   TH3();
   TH3(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                  ,Int_t nbinsy,Double_t ylow,Double_t yup
                                  ,Int_t nbinsz,Double_t zlow,Double_t zup);
   TH3(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                         ,Int_t nbinsy,const Float_t *ybins
                                         ,Int_t nbinsz,const Float_t *zbins);
   TH3(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                         ,Int_t nbinsy,const Double_t *ybins
                                         ,Int_t nbinsz,const Double_t *zbins);
   virtual Int_t    BufferFill(Double_t x, Double_t y, Double_t z, Double_t w);

   void DoFillProfileProjection(TProfile2D * p2, const TAxis & a1, const TAxis & a2, const TAxis & a3, Int_t bin1, Int_t bin2, Int_t bin3, Int_t inBin, Bool_t useWeights) const;

           Int_t    BufferFill(Double_t, Double_t) override {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t, Double_t, Double_t) {return -2;} //may not use
           Int_t    Fill(Double_t) override;        //MayNotUse
           Int_t    Fill(Double_t,Double_t) override {return Fill(0.);} //MayNotUse
           Int_t    Fill(const char*, Double_t) override {return Fill(0);} //MayNotUse
           Int_t    Fill(Double_t,const char*,Double_t) {return Fill(0);} //MayNotUse
           Int_t    Fill(const char*,Double_t,Double_t)  {return Fill(0);} //MayNotUse
           Int_t    Fill(const char*,const char*,Double_t) {return Fill(0);} //MayNotUse

           Double_t Interpolate(Double_t x, Double_t y) const override; // May not use
           Double_t Interpolate(Double_t x) const override; // MayNotUse

private:

   TH3(const TH3&) = delete;
   TH3& operator=(const TH3&) = delete;

   using TH1::Integral;
   using TH1::IntegralAndError;

public:
   virtual ~TH3();
           Int_t    BufferEmpty(Int_t action = 0) override;
           void     Copy(TObject &hnew) const override;
   virtual Int_t    Fill(Double_t x, Double_t y, Double_t z);
   virtual Int_t    Fill(Double_t x, Double_t y, Double_t z, Double_t w);

   virtual Int_t    Fill(const char *namex, const char *namey, const char *namez, Double_t w);
   virtual Int_t    Fill(const char *namex, Double_t y, const char *namez, Double_t w);
   virtual Int_t    Fill(const char *namex, const char *namey, Double_t z, Double_t w);
   virtual Int_t    Fill(Double_t x, const char *namey, const char *namez, Double_t w);
   virtual Int_t    Fill(const char *namex, Double_t y, Double_t z, Double_t w);
   virtual Int_t    Fill(Double_t x, const char *namey, Double_t z, Double_t w);
   virtual Int_t    Fill(Double_t x, Double_t y, const char *namez, Double_t w);

           void     FillRandom(const char *fname, Int_t ntimes=5000, TRandom *rng = nullptr) override;
           void     FillRandom(TH1 *h, Int_t ntimes=5000, TRandom *rng = nullptr) override;
   virtual void     FitSlicesZ(TF1 *f1=0,Int_t binminx=1, Int_t binmaxx=0,Int_t binminy=1, Int_t binmaxy=0,
                                        Int_t cut=0 ,Option_t *option="QNR"); // *MENU*
           Int_t    GetBin(Int_t binx, Int_t biny, Int_t binz) const override;
   using TH1::GetBinContent;
           Double_t GetBinContent(Int_t binx, Int_t biny, Int_t binz) const override { return TH1::GetBinContent( GetBin(binx, biny, binz) ); }
   using TH1::GetBinErrorLow;
   using TH1::GetBinErrorUp;
   virtual Double_t GetBinErrorLow(Int_t binx, Int_t biny, Int_t binz) { return TH1::GetBinErrorLow( GetBin(binx, biny, binz) ); }
   virtual Double_t GetBinErrorUp(Int_t binx, Int_t biny, Int_t binz)  { return TH1::GetBinErrorUp( GetBin(binx, biny, binz) ); }
   virtual Double_t GetBinWithContent3(Double_t c, Int_t &binx, Int_t &biny, Int_t &binz, Int_t firstx=0, Int_t lastx=0,Int_t firsty=0, Int_t lasty=0, Int_t firstz=0, Int_t lastz=0, Double_t maxdiff=0) const;
   virtual Double_t GetCorrelationFactor(Int_t axis1=1,Int_t axis2=2) const;
   virtual Double_t GetCovariance(Int_t axis1=1,Int_t axis2=2) const;
   virtual void     GetRandom3(Double_t &x, Double_t &y, Double_t &, TRandom * rng = nullptr);
           void     GetStats(Double_t *stats) const override;
           Double_t Integral(Option_t *option="") const override;
   virtual Double_t Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Int_t binz1, Int_t binz2, Option_t *option="") const;
   virtual Double_t IntegralAndError(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Int_t binz1, Int_t binz2, Double_t & err, Option_t *option="") const;
           Double_t Interpolate(Double_t x, Double_t y, Double_t z) const override;
           Double_t KolmogorovTest(const TH1 *h2, Option_t *option="") const override;
   virtual TH1D    *ProjectionX(const char *name="_px", Int_t iymin=0, Int_t iymax=-1, Int_t izmin=0,
                                Int_t izmax=-1, Option_t *option="") const; // *MENU*
   virtual TH1D    *ProjectionY(const char *name="_py", Int_t ixmin=0, Int_t ixmax=-1, Int_t izmin=0,
                                Int_t izmax=-1, Option_t *option="") const; // *MENU*
   virtual TH1D    *ProjectionZ(const char *name="_pz", Int_t ixmin=0, Int_t ixmax=-1, Int_t iymin=0,
                                Int_t iymax=-1, Option_t *option="") const; // *MENU*
   virtual TH1     *Project3D(Option_t *option="x") const; // *MENU*
   virtual TProfile2D  *Project3DProfile(Option_t *option="xy") const; // *MENU*
           void     PutStats(Double_t *stats) override;
           TH3     *RebinX(Int_t ngroup = 2, const char *newname = "") override;
   virtual TH3     *RebinY(Int_t ngroup = 2, const char *newname = "");
   virtual TH3     *RebinZ(Int_t ngroup = 2, const char *newname = "");
   virtual TH3     *Rebin3D(Int_t nxgroup = 2, Int_t nygroup = 2, Int_t nzgroup = 2, const char *newname = "");
           void     Reset(Option_t *option="") override;
           void     SetBinContent(Int_t bin, Double_t content) override;
           void     SetBinContent(Int_t bin, Int_t, Double_t content) override { SetBinContent(bin, content); }
           void     SetBinContent(Int_t binx, Int_t biny, Int_t binz, Double_t content) override { SetBinContent(GetBin(binx, biny, binz), content); }
   virtual void     SetShowProjection(const char *option="xy",Int_t nbins=1);   // *MENU*

protected:

   virtual TH1D    *DoProject1D(const char* name, const char * title, int imin1, int imax1, int imin2, int imax2,
                                const TAxis* projAxis, const TAxis * axis1, const TAxis * axis2, Option_t * option) const;
   virtual TH1D    *DoProject1D(const char *name, const char *title, const TAxis *projAxis, const TAxis *axis1,
                                const TAxis *axis2, bool computeErrors, bool originalRange, bool useUF, bool useOF) const;
   virtual TH2D    *DoProject2D(const char* name, const char * title, const TAxis* projX, const TAxis* projY,
                                bool computeErrors, bool originalRange, bool useUF, bool useOF) const;
   virtual TProfile2D *DoProjectProfile2D(const char* name, const char * title, const TAxis* projX, const TAxis* projY,
                                           bool originalRange, bool useUF, bool useOF) const;

   // these functions are need to be used inside TProfile3D::DoProjectProfile2D
   static TH1D     *DoProject1D(const TH3 & h, const char* name, const char * title, const TAxis* projX,
                                bool computeErrors, bool originalRange, bool useUF, bool useOF);
   static TH2D     *DoProject2D(const TH3 & h, const char* name, const char * title, const TAxis* projX, const TAxis* projY,
                                bool computeErrors, bool originalRange, bool useUF, bool useOF);

   ClassDefOverride(TH3,6)  //3-Dim histogram base class
};

//________________________________________________________________________

class TH3C : public TH3, public TArrayC {
public:
   TH3C();
   TH3C(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                  ,Int_t nbinsy,Double_t ylow,Double_t yup
                                  ,Int_t nbinsz,Double_t zlow,Double_t zup);
   TH3C(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3C(const TH3C &h3c);
   virtual ~TH3C();

           void      AddBinContent(Int_t bin) override;
           void      AddBinContent(Int_t bin, Double_t w) override;
           void      Copy(TObject &hnew) const override;
           void      Reset(Option_t *option="") override;
           void      SetBinsLength(Int_t n=-1) override;

           TH3C&     operator=(const TH3C &h1);
   friend  TH3C      operator*(Float_t c1, TH3C &h1);
   friend  TH3C      operator*(TH3C &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3C      operator+(TH3C &h1, TH3C &h2);
   friend  TH3C      operator-(TH3C &h1, TH3C &h2);
   friend  TH3C      operator*(TH3C &h1, TH3C &h2);
   friend  TH3C      operator/(TH3C &h1, TH3C &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Char_t (content); }

   ClassDefOverride(TH3C,4)  //3-Dim histograms (one char per channel)
};

//________________________________________________________________________

class TH3S : public TH3, public TArrayS {
public:
   TH3S();
   TH3S(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                  ,Int_t nbinsy,Double_t ylow,Double_t yup
                                  ,Int_t nbinsz,Double_t zlow,Double_t zup);
   TH3S(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3S(const TH3S &h3s);
   virtual ~TH3S();

           void      AddBinContent(Int_t bin) override;
           void      AddBinContent(Int_t bin, Double_t w) override;
           void      Copy(TObject &hnew) const override;
           void      Reset(Option_t *option="") override;
           void      SetBinsLength(Int_t n=-1) override;

           TH3S&     operator=(const TH3S &h1);
   friend  TH3S      operator*(Float_t c1, TH3S &h1);
   friend  TH3S      operator*(TH3S &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3S      operator+(TH3S &h1, TH3S &h2);
   friend  TH3S      operator-(TH3S &h1, TH3S &h2);
   friend  TH3S      operator*(TH3S &h1, TH3S &h2);
   friend  TH3S      operator/(TH3S &h1, TH3S &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Short_t (content); }

   ClassDefOverride(TH3S,4)  //3-Dim histograms (one short per channel)
};

//________________________________________________________________________

class TH3I : public TH3, public TArrayI {
public:
   TH3I();
   TH3I(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                  ,Int_t nbinsy,Double_t ylow,Double_t yup
                                  ,Int_t nbinsz,Double_t zlow,Double_t zup);
   TH3I(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3I(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3I(const TH3I &h3i);
   virtual ~TH3I();

           void      AddBinContent(Int_t bin) override;
           void      AddBinContent(Int_t bin, Double_t w) override;
           void      Copy(TObject &hnew) const override;
           void      Reset(Option_t *option="") override;
           void      SetBinsLength(Int_t n=-1) override;

           TH3I&     operator=(const TH3I &h1);
   friend  TH3I      operator*(Float_t c1, TH3I &h1);
   friend  TH3I      operator*(TH3I &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3I      operator+(TH3I &h1, TH3I &h2);
   friend  TH3I      operator-(TH3I &h1, TH3I &h2);
   friend  TH3I      operator*(TH3I &h1, TH3I &h2);
   friend  TH3I      operator/(TH3I &h1, TH3I &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Int_t (content); }

   ClassDefOverride(TH3I,4)  //3-Dim histograms (one 32 bits integer per channel)
};


//________________________________________________________________________

class TH3F : public TH3, public TArrayF {
public:
   TH3F();
   TH3F(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                  ,Int_t nbinsy,Double_t ylow,Double_t yup
                                  ,Int_t nbinsz,Double_t zlow,Double_t zup);
   TH3F(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3F(const TH3F &h3f);
   virtual ~TH3F();

           void      AddBinContent(Int_t bin) override {++fArray[bin];}
           void      AddBinContent(Int_t bin, Double_t w) override
                                 {fArray[bin] += Float_t (w);}
           void      Copy(TObject &hnew) const override;
           void      Reset(Option_t *option="") override;
           void      SetBinsLength(Int_t n=-1) override;

           TH3F&     operator=(const TH3F &h1);
   friend  TH3F      operator*(Float_t c1, TH3F &h1);
   friend  TH3F      operator*(TH3F &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3F      operator+(TH3F &h1, TH3F &h2);
   friend  TH3F      operator-(TH3F &h1, TH3F &h2);
   friend  TH3F      operator*(TH3F &h1, TH3F &h2);
   friend  TH3F      operator/(TH3F &h1, TH3F &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return Double_t (fArray[bin]); }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = Float_t (content); }

   ClassDefOverride(TH3F,4)  //3-Dim histograms (one float per channel)
};

//________________________________________________________________________

class TH3D : public TH3, public TArrayD {
public:
   TH3D();
   TH3D(const char *name,const char *title,Int_t nbinsx,Double_t xlow,Double_t xup
                                  ,Int_t nbinsy,Double_t ylow,Double_t yup
                                  ,Int_t nbinsz,Double_t zlow,Double_t zup);
   TH3D(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3D(const TH3D &h3d);
   virtual ~TH3D();

           void      AddBinContent(Int_t bin) override {++fArray[bin];}
           void      AddBinContent(Int_t bin, Double_t w) override
                                 {fArray[bin] += Double_t (w);}
           void      Copy(TObject &hnew) const override;
           void      Reset(Option_t *option="") override;
           void      SetBinsLength(Int_t n=-1) override;

           TH3D&     operator=(const TH3D &h1);
   friend  TH3D      operator*(Float_t c1, TH3D &h1);
   friend  TH3D      operator*(TH3D &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3D      operator+(TH3D &h1, TH3D &h2);
   friend  TH3D      operator-(TH3D &h1, TH3D &h2);
   friend  TH3D      operator*(TH3D &h1, TH3D &h2);
   friend  TH3D      operator/(TH3D &h1, TH3D &h2);

protected:
           Double_t RetrieveBinContent(Int_t bin) const override { return fArray[bin]; }
           void     UpdateBinContent(Int_t bin, Double_t content) override { fArray[bin] = content; }

   ClassDefOverride(TH3D,4)  //3-Dim histograms (one double per channel)
};

#endif
