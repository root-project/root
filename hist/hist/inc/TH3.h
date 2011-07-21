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

#ifndef ROOT_TH1
#include "TH1.h"
#endif

#ifndef ROOT_TAtt3D
#include "TAtt3D.h"
#endif

class TH2D; 
class TProfile2D;

class TH3 : public TH1, public TAtt3D {

protected:
   Double_t     fTsumwy;          //Total Sum of weight*Y
   Double_t     fTsumwy2;         //Total Sum of weight*Y*Y
   Double_t     fTsumwxy;         //Total Sum of weight*X*Y
   Double_t     fTsumwz;          //Total Sum of weight*Z
   Double_t     fTsumwz2;         //Total Sum of weight*Z*Z
   Double_t     fTsumwxz;         //Total Sum of weight*X*Z
   Double_t     fTsumwyz;         //Total Sum of weight*Y*Z

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
   virtual Int_t    BufferFill(Double_t, Double_t) {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t, Double_t, Double_t) {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t x, Double_t y, Double_t z, Double_t w);

   void DoFillProfileProjection(TProfile2D * p2, const TAxis & a1, const TAxis & a2, const TAxis & a3, Int_t bin1, Int_t bin2, Int_t bin3, Int_t inBin, Bool_t useWeights) const;

public:
   TH3(const TH3&);
   virtual ~TH3();
   
   virtual Int_t    BufferEmpty(Int_t action=0);
   virtual void     Copy(TObject &hnew) const;
           Int_t    Fill(Double_t) {return -1;}        //MayNotUse
           Int_t    Fill(Double_t,Double_t) {return -1;} //MayNotUse
           Int_t    Fill(const char*, Double_t) {return -1;} //MayNotUse
           Int_t    Fill(Double_t,const char*,Double_t) {return -1;} //MayNotUse
           Int_t    Fill(const char*,Double_t,Double_t) {return -1;} //MayNotUse
           Int_t    Fill(const char*,const char*,Double_t) {return -1;} //MayNotUse
   virtual Int_t    Fill(Double_t x, Double_t y, Double_t z);
   virtual Int_t    Fill(Double_t x, Double_t y, Double_t z, Double_t w);

   virtual Int_t    Fill(const char *namex, const char *namey, const char *namez, Double_t w);
   virtual Int_t    Fill(const char *namex, Double_t y, const char *namez, Double_t w);
   virtual Int_t    Fill(const char *namex, const char *namey, Double_t z, Double_t w);
   virtual Int_t    Fill(Double_t x, const char *namey, const char *namez, Double_t w);
   virtual Int_t    Fill(Double_t x, const char *namey, Double_t z, Double_t w);
   virtual Int_t    Fill(Double_t x, Double_t y, const char *namez, Double_t w);

   virtual void     FillRandom(const char *fname, Int_t ntimes=5000);
   virtual void     FillRandom(TH1 *h, Int_t ntimes=5000);
   virtual Int_t    FindFirstBinAbove(Double_t threshold=0, Int_t axis=1) const;
   virtual Int_t    FindLastBinAbove (Double_t threshold=0, Int_t axis=1) const;
   virtual void     FitSlicesZ(TF1 *f1=0,Int_t binminx=1, Int_t binmaxx=0,Int_t binminy=1, Int_t binmaxy=0,
                                        Int_t cut=0 ,Option_t *option="QNR"); // *MENU*
   virtual Double_t GetBinWithContent3(Double_t c, Int_t &binx, Int_t &biny, Int_t &binz, Int_t firstx=0, Int_t lastx=0,Int_t firsty=0, Int_t lasty=0, Int_t firstz=0, Int_t lastz=0, Double_t maxdiff=0) const;
   virtual Double_t GetCorrelationFactor(Int_t axis1=1,Int_t axis2=2) const;
   virtual Double_t GetCovariance(Int_t axis1=1,Int_t axis2=2) const;
   virtual void     GetRandom3(Double_t &x, Double_t &y, Double_t &z);
   virtual void     GetStats(Double_t *stats) const;
   virtual Double_t Integral(Option_t *option="") const;
   using TH1::Integral;
   virtual Double_t Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Int_t binz1, Int_t binz2, Option_t *option="") const;
   using TH1::IntegralAndError;
   virtual Double_t IntegralAndError(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Int_t binz1, Int_t binz2, Double_t & err, Option_t *option="") const;
   virtual Double_t Interpolate(Double_t x);
   virtual Double_t Interpolate(Double_t x, Double_t y);
   virtual Double_t Interpolate(Double_t x, Double_t y, Double_t z);
   virtual Double_t KolmogorovTest(const TH1 *h2, Option_t *option="") const;
   virtual Long64_t Merge(TCollection *list);
         TH1D      *ProjectionX(const char *name="_px", Int_t firstybin=0, Int_t lastybin=-1, Int_t firstzbin=0,
                                Int_t lastzbin=-1, Option_t *option="") const; // *MENU*
         TH1D      *ProjectionY(const char *name="_py", Int_t firstxbin=0, Int_t lastxbin=-1, Int_t firstzbin=0,
                                Int_t lastzbin=-1, Option_t *option="") const; // *MENU*
         TH1D      *ProjectionZ(const char *name="_pz", Int_t firstxbin=0, Int_t lastxbin=-1, Int_t firstybin=0,
                                Int_t lastybin=-1, Option_t *option="") const; // *MENU*
         TH1       *Project3D(Option_t *option="x") const; // *MENU*
   TProfile2D      *Project3DProfile(Option_t *option="xy") const; // *MENU*
   virtual void     PutStats(Double_t *stats);
   virtual TH3     *RebinX(Int_t ngroup, const char *newname);
   virtual TH3     *RebinY(Int_t ngroup, const char *newname);
   virtual TH3     *RebinZ(Int_t ngroup, const char *newname);
   virtual TH3     *Rebin3D(Int_t nxgroup, Int_t nygroup, Int_t nzgroup, const char *newname);
   virtual void     Reset(Option_t *option="");
   virtual void     SetShowProjection(const char *option="xy",Int_t nbins=1);   // *MENU*

protected:
   TH1D        *DoProject1D(const char* name, const char * title, TAxis* projX, 
                         bool computeErrors, bool originalRange,
                         bool useUF, bool useOF) const;
   TH2D        *DoProject2D(const char* name, const char * title, TAxis* projX, TAxis* projY, 
                        bool computeErrors, bool originalRange,
                         bool useUF, bool useOF) const;
   TProfile2D *DoProjectProfile2D(const char* name, const char * title, TAxis* projX, TAxis* projY, 
                                        bool originalRange, bool useUF, bool useOF) const;
   
   ClassDef(TH3,5)  //3-Dim histogram base class
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
   virtual void      AddBinContent(Int_t bin);
   virtual void      AddBinContent(Int_t bin, Double_t w);
   virtual void      Copy(TObject &hnew) const;
   virtual TH1      *DrawCopy(Option_t *option="") const ;
   virtual Double_t  GetBinContent(Int_t bin) const;
   virtual Double_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void      Reset(Option_t *option="");
   virtual void      SetBinContent(Int_t bin, Double_t content);
   virtual void      SetBinContent(Int_t bin, Int_t, Double_t content) {SetBinContent(bin,content);}
   virtual void      SetBinContent(Int_t binx, Int_t biny, Int_t binz, Double_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void      SetBinsLength(Int_t n=-1);
           TH3C&     operator=(const TH3C &h1);
   friend  TH3C      operator*(Float_t c1, TH3C &h1);
   friend  TH3C      operator*(TH3C &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3C      operator+(TH3C &h1, TH3C &h2);
   friend  TH3C      operator-(TH3C &h1, TH3C &h2);
   friend  TH3C      operator*(TH3C &h1, TH3C &h2);
   friend  TH3C      operator/(TH3C &h1, TH3C &h2);

   ClassDef(TH3C,3)  //3-Dim histograms (one char per channel)
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
   virtual void      AddBinContent(Int_t bin);
   virtual void      AddBinContent(Int_t bin, Double_t w);
   virtual void      Copy(TObject &hnew) const;
   virtual TH1      *DrawCopy(Option_t *option="") const;
   virtual Double_t  GetBinContent(Int_t bin) const;
   virtual Double_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void      Reset(Option_t *option="");
   virtual void      SetBinContent(Int_t bin, Double_t content);
   virtual void      SetBinContent(Int_t bin, Int_t, Double_t content) {SetBinContent(bin,content);}
   virtual void      SetBinContent(Int_t binx, Int_t biny, Int_t binz, Double_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void      SetBinsLength(Int_t n=-1);
           TH3S&     operator=(const TH3S &h1);
   friend  TH3S      operator*(Float_t c1, TH3S &h1);
   friend  TH3S      operator*(TH3S &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3S      operator+(TH3S &h1, TH3S &h2);
   friend  TH3S      operator-(TH3S &h1, TH3S &h2);
   friend  TH3S      operator*(TH3S &h1, TH3S &h2);
   friend  TH3S      operator/(TH3S &h1, TH3S &h2);

   ClassDef(TH3S,3)  //3-Dim histograms (one short per channel)
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
   virtual void      AddBinContent(Int_t bin);
   virtual void      AddBinContent(Int_t bin, Double_t w);
   virtual void      Copy(TObject &hnew) const;
   virtual TH1      *DrawCopy(Option_t *option="") const;
   virtual Double_t  GetBinContent(Int_t bin) const;
   virtual Double_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void      Reset(Option_t *option="");
   virtual void      SetBinContent(Int_t bin, Double_t content);
   virtual void      SetBinContent(Int_t bin, Int_t, Double_t content) {SetBinContent(bin,content);}
   virtual void      SetBinContent(Int_t binx, Int_t biny, Int_t binz, Double_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void      SetBinsLength(Int_t n=-1);
           TH3I&     operator=(const TH3I &h1);
   friend  TH3I      operator*(Float_t c1, TH3I &h1);
   friend  TH3I      operator*(TH3I &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3I      operator+(TH3I &h1, TH3I &h2);
   friend  TH3I      operator-(TH3I &h1, TH3I &h2);
   friend  TH3I      operator*(TH3I &h1, TH3I &h2);
   friend  TH3I      operator/(TH3I &h1, TH3I &h2);

   ClassDef(TH3I,3)  //3-Dim histograms (one 32 bits integer per channel)
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
   virtual void      AddBinContent(Int_t bin);
   virtual void      AddBinContent(Int_t bin, Double_t w);
   virtual void      Copy(TObject &hnew) const;
   virtual TH1      *DrawCopy(Option_t *option="") const;
   virtual Double_t  GetBinContent(Int_t bin) const;
   virtual Double_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void      Reset(Option_t *option="");
   virtual void      SetBinContent(Int_t bin, Double_t content);
   virtual void      SetBinContent(Int_t bin, Int_t, Double_t content) {SetBinContent(bin,content);}
   virtual void      SetBinContent(Int_t binx, Int_t biny, Int_t binz, Double_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void      SetBinsLength(Int_t n=-1);
           TH3F&     operator=(const TH3F &h1);
   friend  TH3F      operator*(Float_t c1, TH3F &h1);
   friend  TH3F      operator*(TH3F &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3F      operator+(TH3F &h1, TH3F &h2);
   friend  TH3F      operator-(TH3F &h1, TH3F &h2);
   friend  TH3F      operator*(TH3F &h1, TH3F &h2);
   friend  TH3F      operator/(TH3F &h1, TH3F &h2);

   ClassDef(TH3F,3)  //3-Dim histograms (one float per channel)
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
   virtual void      AddBinContent(Int_t bin);
   virtual void      AddBinContent(Int_t bin, Double_t w);
   virtual void      Copy(TObject &hnew) const;
   virtual TH1      *DrawCopy(Option_t *option="") const;
   virtual Double_t  GetBinContent(Int_t bin) const;
   virtual Double_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void      Reset(Option_t *option="");
   virtual void      SetBinContent(Int_t bin, Double_t content);
   virtual void      SetBinContent(Int_t bin, Int_t, Double_t content) {SetBinContent(bin,content);}
   virtual void      SetBinContent(Int_t binx, Int_t biny, Int_t binz, Double_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void      SetBinsLength(Int_t n=-1);
           TH3D&     operator=(const TH3D &h1);
   friend  TH3D      operator*(Float_t c1, TH3D &h1);
   friend  TH3D      operator*(TH3D &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3D      operator+(TH3D &h1, TH3D &h2);
   friend  TH3D      operator-(TH3D &h1, TH3D &h2);
   friend  TH3D      operator*(TH3D &h1, TH3D &h2);
   friend  TH3D      operator/(TH3D &h1, TH3D &h2);

   ClassDef(TH3D,3)  //3-Dim histograms (one double per channel)
};

#endif

