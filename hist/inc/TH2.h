// @(#)root/hist:$Name:  $:$Id: TH2.h,v 1.9 2001/01/08 08:34:27 brun Exp $
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

#ifndef ROOT_TH1
#include "TH1.h"
#endif

class TProfile;
class TMatrix;
class TMatrixD;

class TH2 : public TH1 {

protected:
   Stat_t     fScalefactor;     //Scale factor
   Stat_t     fTsumwy;          //Total Sum of weight*Y
   Stat_t     fTsumwy2;         //Total Sum of weight*Y*Y
   Stat_t     fTsumwxy;         //Total Sum of weight*X*Y

public:
   TH2();
   TH2(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                         ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                         ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                         ,Int_t nbinsy,Double_t *ybins);
   TH2(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                         ,Int_t nbinsy,Double_t *ybins);
   TH2(const char *name,const char *title,Int_t nbinsx,Float_t  *xbins
                                         ,Int_t nbinsy,Float_t  *ybins);
   virtual ~TH2();
   virtual void    Copy(TObject &hnew);
           Int_t   Fill(Axis_t) {return -1;} //MayNotUse
   virtual Int_t   Fill(Axis_t x, Axis_t y);
   virtual Int_t   Fill(Axis_t x, Axis_t y, Stat_t w);
   virtual void    FillN(Int_t, Axis_t *, Double_t *, Int_t) {;} //MayNotUse
   virtual void    FillN(Int_t ntimes, Axis_t *x, Axis_t *y, Double_t *w, Int_t stride=1);
   virtual void    FillRandom(const char *fname, Int_t ntimes=5000);
   virtual void    FillRandom(TH1 *h, Int_t ntimes=5000);
   virtual void    FitSlicesX(TF1 *f1=0,Int_t binmin=1, Int_t binmax=0, Int_t cut=0 ,Option_t *option="QNR"); // *MENU*
   virtual void    FitSlicesY(TF1 *f1=0,Int_t binmin=1, Int_t binmax=0, Int_t cut=0 ,Option_t *option="QNR"); // *MENU*
   virtual Stat_t  GetCorrelationFactor(Int_t axis1=1,Int_t axis2=2) const;
   virtual Stat_t  GetCovariance(Int_t axis1=1,Int_t axis2=2) const;
   virtual void    GetRandom2(Axis_t &x, Axis_t &y);
   virtual void    GetStats(Stat_t *stats) const;
   virtual Stat_t  Integral();
   virtual Stat_t  Integral(Int_t, Int_t) {return 0;}
   virtual Stat_t  Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2);
   virtual Stat_t  Integral(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t) {return 0;}
   virtual Double_t KolmogorovTest(TH1 *h2, Option_t *option="");
      TProfile    *ProfileX(const char *name="_pfx", Int_t firstybin=0, Int_t lastybin=9999, Option_t *option="");   // *MENU*
      TProfile    *ProfileY(const char *name="_pfy", Int_t firstxbin=0, Int_t lastxbin=9999, Option_t *option="");   // *MENU*
          TH1D    *ProjectionX(const char *name="_px", Int_t firstybin=0, Int_t lastybin=9999, Option_t *option=""); // *MENU*
          TH1D    *ProjectionY(const char *name="_py", Int_t firstxbin=0, Int_t lastxbin=9999, Option_t *option=""); // *MENU*
   virtual void    PutStats(Stat_t *stats);
   virtual void    Reset(Option_t *option="");

   ClassDef(TH2,3)  //2-Dim histogram base class
};


//______________________________________________________________________________

class TH2C : public TH2, public TArrayC {

public:
   TH2C();
   TH2C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2C(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2C(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2C(const char *name,const char *title,Int_t nbinsx,Float_t  *xbins
                                          ,Int_t nbinsy,Float_t  *ybins);
   TH2C(const TH2C &h2c);
   virtual ~TH2C();
   virtual void    AddBinContent(Int_t bin);
   virtual void    AddBinContent(Int_t bin, Stat_t w);
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny) const {return GetBinContent(GetBin(binx,biny));}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const {return GetBinContent(GetBin(binx,biny));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Char_t (content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayC::Set(nx);}
           TH2C&   operator=(const TH2C &h1);
   friend  TH2C    operator*(Float_t c1, TH2C &h1);
   friend  TH2C    operator*(TH2C &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2C    operator+(TH2C &h1, TH2C &h2);
   friend  TH2C    operator-(TH2C &h1, TH2C &h2);
   friend  TH2C    operator*(TH2C &h1, TH2C &h2);
   friend  TH2C    operator/(TH2C &h1, TH2C &h2);

   ClassDef(TH2C,3)  //2-Dim histograms (one char per channel)
};


//______________________________________________________________________________

class TH2S : public TH2, public TArrayS {

public:
   TH2S();
   TH2S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2S(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2S(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2S(const char *name,const char *title,Int_t nbinsx,Float_t  *xbins
                                          ,Int_t nbinsy,Float_t  *ybins);
   TH2S(const TH2S &h2s);
   virtual ~TH2S();
   virtual void    AddBinContent(Int_t bin);
   virtual void    AddBinContent(Int_t bin, Stat_t w);
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny) const {return GetBinContent(GetBin(binx,biny));}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const {return GetBinContent(GetBin(binx,biny));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Short_t (content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayS::Set(nx);}
           TH2S&   operator=(const TH2S &h1);
   friend  TH2S    operator*(Float_t c1, TH2S &h1);
   friend  TH2S    operator*(TH2S &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2S    operator+(TH2S &h1, TH2S &h2);
   friend  TH2S    operator-(TH2S &h1, TH2S &h2);
   friend  TH2S    operator*(TH2S &h1, TH2S &h2);
   friend  TH2S    operator/(TH2S &h1, TH2S &h2);

   ClassDef(TH2S,3)  //2-Dim histograms (one short per channel)
};


//______________________________________________________________________________

class TH2F : public TH2, public TArrayF {

public:
   TH2F();
   TH2F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2F(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2F(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2F(const char *name,const char *title,Int_t nbinsx,Float_t  *xbins
                                          ,Int_t nbinsy,Float_t  *ybins);
   TH2F(const TMatrix &m);
   TH2F(const TH2F &h2f);
   virtual ~TH2F();
   virtual void    AddBinContent(Int_t bin) {++fArray[bin];}
   virtual void    AddBinContent(Int_t bin, Stat_t w)
                                 {fArray[bin] += Float_t (w);}
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny) const {return GetBinContent(GetBin(binx,biny));}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const {return GetBinContent(GetBin(binx,biny));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Float_t (content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayF::Set(nx);}
           TH2F&   operator=(const TH2F &h1);
   friend  TH2F    operator*(Float_t c1, TH2F &h1);
   friend  TH2F    operator*(TH2F &h1, Float_t c1);
   friend  TH2F    operator+(TH2F &h1, TH2F &h2);
   friend  TH2F    operator-(TH2F &h1, TH2F &h2);
   friend  TH2F    operator*(TH2F &h1, TH2F &h2);
   friend  TH2F    operator/(TH2F &h1, TH2F &h2);

   ClassDef(TH2F,3)  //2-Dim histograms (one float per channel)
};


//______________________________________________________________________________

class TH2D : public TH2, public TArrayD {

public:
   TH2D();
   TH2D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2D(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Axis_t ylow,Axis_t yup);
   TH2D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2D(const char *name,const char *title,Int_t nbinsx,Double_t *xbins
                                          ,Int_t nbinsy,Double_t *ybins);
   TH2D(const char *name,const char *title,Int_t nbinsx,Float_t  *xbins
                                          ,Int_t nbinsy,Float_t  *ybins);
   TH2D(const TMatrixD &m);
   TH2D(const TH2D &h2d);
   virtual ~TH2D();
   virtual void    AddBinContent(Int_t bin) {++fArray[bin];}
   virtual void    AddBinContent(Int_t bin, Stat_t w)
                                 {fArray[bin] += Double_t (w);}
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny) const {return GetBinContent(GetBin(binx,biny));}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const {return GetBinContent(GetBin(binx,biny));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Double_t (content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t, Stat_t content) {SetBinContent(GetBin(binx,biny),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayD::Set(nx);}
           TH2D&   operator=(const TH2D &h1);
   friend  TH2D    operator*(Float_t c1, TH2D &h1);
   friend  TH2D    operator*(TH2D &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH2D    operator+(TH2D &h1, TH2D &h2);
   friend  TH2D    operator-(TH2D &h1, TH2D &h2);
   friend  TH2D    operator*(TH2D &h1, TH2D &h2);
   friend  TH2D    operator/(TH2D &h1, TH2D &h2);

   ClassDef(TH2D,3)  //2-Dim histograms (one double per channel)
};

#endif

