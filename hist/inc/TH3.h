// @(#)root/hist:$Name:  $:$Id: TH3.h,v 1.11 2001/02/13 11:57:30 brun Exp $
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


class TH3 : public TH1, public TAtt3D {

public:
   TH3();
   TH3(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                  ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                  ,Int_t nbinsz,Axis_t zlow,Axis_t zup);
   TH3(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                         ,Int_t nbinsy,const Float_t *ybins
                                         ,Int_t nbinsz,const Float_t *zbins);
   TH3(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                         ,Int_t nbinsy,const Double_t *ybins
                                         ,Int_t nbinsz,const Double_t *zbins);
   virtual ~TH3();
   virtual void    Copy(TObject &hnew);
           Int_t   Fill(Axis_t) {return -1;}        //MayNotUse
           Int_t   Fill(Axis_t,Stat_t) {return -1;} //MayNotUse
   virtual Int_t   Fill(Axis_t x, Axis_t y, Axis_t z);
   virtual Int_t   Fill(Axis_t x, Axis_t y, Axis_t z, Stat_t w);
   virtual void    FillRandom(const char *fname, Int_t ntimes=5000);
   virtual void    FillRandom(TH1 *h, Int_t ntimes=5000);
   virtual void    FitSlicesZ(TF1 *f1=0,Int_t binminx=1, Int_t binmaxx=0,Int_t binminy=1, Int_t binmaxy=0,
                                        Int_t cut=0 ,Option_t *option="QNR"); // *MENU*
   virtual void    GetRandom3(Axis_t &x, Axis_t &y, Axis_t &z);
   virtual void    GetStats(Stat_t *stats) const;
   virtual Stat_t  Integral(Option_t *option="");
   virtual Stat_t  Integral(Int_t, Int_t, Option_t *option="") {return 0;}
   virtual Stat_t  Integral(Int_t, Int_t, Int_t, Int_t, Option_t *option="") {return 0;}
   virtual Stat_t  Integral(Int_t binx1, Int_t binx2, Int_t biny1, Int_t biny2, Int_t binz1, Int_t binz2, Option_t *option="");
   virtual Double_t KolmogorovTest(TH1 *h2, Option_t *option="");
          TH1D    *ProjectionZ(const char *name="_pz", Int_t firstxbin=0, Int_t lastxbin=9999, Int_t firstybin=0,
                                 Int_t lastybin=9999, Option_t *option=""); // *MENU*
          TH1     *Project3D(Option_t *option="x"); // *MENU*
   virtual void    PutStats(Stat_t *stats);
   virtual void    Reset(Option_t *option="");
   virtual void    Sizeof3D() const;

   ClassDef(TH3,3)  //3-Dim histogram base class
};

//________________________________________________________________________

class TH3C : public TH3, public TArrayC {
public:
   TH3C();
   TH3C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                  ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                  ,Int_t nbinsz,Axis_t zlow,Axis_t zup);
   TH3C(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3C(const TH3C &h3c);
   virtual ~TH3C();
   virtual void    AddBinContent(Int_t bin);
   virtual void    AddBinContent(Int_t bin, Stat_t w);
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Char_t (content);}
   virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t binz, Stat_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayC::Set(nx);}
           TH3C&   operator=(const TH3C &h1);
   friend  TH3C    operator*(Float_t c1, TH3C &h1);
   friend  TH3C    operator*(TH3C &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3C    operator+(TH3C &h1, TH3C &h2);
   friend  TH3C    operator-(TH3C &h1, TH3C &h2);
   friend  TH3C    operator*(TH3C &h1, TH3C &h2);
   friend  TH3C    operator/(TH3C &h1, TH3C &h2);

   ClassDef(TH3C,3)  //3-Dim histograms (one char per channel)
};

//________________________________________________________________________

class TH3S : public TH3, public TArrayS {
public:
   TH3S();
   TH3S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                  ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                  ,Int_t nbinsz,Axis_t zlow,Axis_t zup);
   TH3S(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3S(const TH3S &h3s);
   virtual ~TH3S();
   virtual void    AddBinContent(Int_t bin);
   virtual void    AddBinContent(Int_t bin, Stat_t w);
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Short_t (content);}
   virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t binz, Stat_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayS::Set(nx);}
           TH3S&   operator=(const TH3S &h1);
   friend  TH3S    operator*(Float_t c1, TH3S &h1);
   friend  TH3S    operator*(TH3S &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3S    operator+(TH3S &h1, TH3S &h2);
   friend  TH3S    operator-(TH3S &h1, TH3S &h2);
   friend  TH3S    operator*(TH3S &h1, TH3S &h2);
   friend  TH3S    operator/(TH3S &h1, TH3S &h2);

   ClassDef(TH3S,3)  //3-Dim histograms (one short per channel)
};


//________________________________________________________________________

class TH3F : public TH3, public TArrayF {
public:
   TH3F();
   TH3F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                  ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                  ,Int_t nbinsz,Axis_t zlow,Axis_t zup);
   TH3F(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3F(const TH3F &h3f);
   virtual ~TH3F();
   virtual void    AddBinContent(Int_t bin) {++fArray[bin];}
   virtual void    AddBinContent(Int_t bin, Stat_t w)
                                 {fArray[bin] += Float_t (w);}
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Float_t (content);}
   virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t binz, Stat_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayF::Set(nx);}
           TH3F&   operator=(const TH3F &h1);
   friend  TH3F    operator*(Float_t c1, TH3F &h1);
   friend  TH3F    operator*(TH3F &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3F    operator+(TH3F &h1, TH3F &h2);
   friend  TH3F    operator-(TH3F &h1, TH3F &h2);
   friend  TH3F    operator*(TH3F &h1, TH3F &h2);
   friend  TH3F    operator/(TH3F &h1, TH3F &h2);

   ClassDef(TH3F,3)  //3-Dim histograms (one float per channel)
};

//________________________________________________________________________

class TH3D : public TH3, public TArrayD {
public:
   TH3D();
   TH3D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                  ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                  ,Int_t nbinsz,Axis_t zlow,Axis_t zup);
   TH3D(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins
                                          ,Int_t nbinsy,const Float_t *ybins
                                          ,Int_t nbinsz,const Float_t *zbins);
   TH3D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins
                                          ,Int_t nbinsy,const Double_t *ybins
                                          ,Int_t nbinsz,const Double_t *zbins);
   TH3D(const TH3D &h3d);
   virtual ~TH3D();
   virtual void    AddBinContent(Int_t bin) {++fArray[bin];}
   virtual void    AddBinContent(Int_t bin, Stat_t w)
                                 {fArray[bin] += Double_t (w);}
   virtual void    Copy(TObject &hnew);
   virtual TH1    *DrawCopy(Option_t *option="");
   virtual Stat_t  GetBinContent(Int_t bin) const;
   virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
   virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t binz) const {return GetBinContent(GetBin(binx,biny,binz));}
   virtual void    Reset(Option_t *option="");
   virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Double_t (content);}
   virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
   virtual void    SetBinContent(Int_t binx, Int_t biny, Int_t binz, Stat_t content) {SetBinContent(GetBin(binx,biny,binz),content);}
   virtual void    SetBinsLength(Int_t nx) {TArrayD::Set(nx);}
           TH3D&   operator=(const TH3D &h1);
   friend  TH3D    operator*(Float_t c1, TH3D &h1);
   friend  TH3D    operator*(TH3D &h1, Float_t c1) {return operator*(c1,h1);}
   friend  TH3D    operator+(TH3D &h1, TH3D &h2);
   friend  TH3D    operator-(TH3D &h1, TH3D &h2);
   friend  TH3D    operator*(TH3D &h1, TH3D &h2);
   friend  TH3D    operator/(TH3D &h1, TH3D &h2);

   ClassDef(TH3D,3)  //3-Dim histograms (one double per channel)
};

#endif

