// @(#)root/hist:$Name:  $:$Id: TProfile.h,v 1.3 2000/06/15 06:51:49 brun Exp $
// Author: Rene Brun   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProfile
#define ROOT_TProfile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProfile                                                             //
//                                                                      //
// Profile histogram class.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TH1
#include "TH1.h"
#endif

enum EErrorType { kERRORMEAN = 0, kERRORSPREAD, kERRORSPREADI, kERRORSPREADG };


class TProfile : public TH1D {

protected:
    TArrayD     fBinEntries;      //number of entries per bin
    EErrorType  fErrorMode;       //Option to compute errors
    Double_t    fYmin;            //Lower limit in Y (if set)
    Double_t    fYmax;            //Upper limit in Y (if set)

private:
   Int_t Fill(Axis_t) { MayNotUse("Fill(Axis_t)"); return -1;}
   void FillN(Int_t, Axis_t *, Double_t *, Int_t) { MayNotUse("FillN(Int_t, Axis_t*, Double_t*, Int_t)"); }
   void SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t)
      { MayNotUse("SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t"); }
   void SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t)
      { MayNotUse("SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t"); }
   Double_t *GetB()  {return &fBinEntries.fArray[0];}
   Double_t *GetW()  {return &fArray[0];}
   Double_t *GetW2() {return &fSumw2.fArray[0];}

public:
    TProfile();
    TProfile(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup, Option_t *option="");
    TProfile(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup,Axis_t ylow,Axis_t yup,Option_t *option="");
    TProfile(const char *name,const char *title,Int_t nbinsx,Float_t  *xbins, Option_t *option="");
    TProfile(const char *name,const char *title,Int_t nbinsx,Double_t *xbins, Option_t *option="");
    TProfile(const TProfile &profile);
    virtual ~TProfile();
    virtual void    Add(TH1 *h1, Double_t c1=1);
    virtual void    Add(TH1 *h1, TH1 *h2, Double_t c1=1, Double_t c2=1); // *MENU*
            void    BuildOptions(Double_t ymin, Double_t ymax, Option_t *option);
    virtual void    Copy(TObject &hnew);
    virtual void    Divide(TH1 *h1);
    virtual void    Divide(TH1 *h1, TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
    virtual TH1    *DrawCopy(Option_t *option="");
    virtual Int_t   Fill(Axis_t x, Axis_t y);
    virtual Int_t   Fill(Axis_t x, Axis_t y, Stat_t w);
    virtual void    FillN(Int_t ntimes, Axis_t *x, Axis_t *y, Double_t *w, Int_t stride=1);
    virtual Stat_t  GetBinContent(Int_t bin);
    virtual Stat_t  GetBinError(Int_t bin);
    virtual Stat_t  GetBinEntries(Int_t bin);
    Option_t       *GetErrorOption() const;
    virtual Double_t GetYmin() {return fYmin;}
    virtual Double_t GetYmax() {return fYmax;}
    virtual void    Multiply(TH1 *h1);
    virtual void    Multiply(TH1 *h1, TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
            TH1D   *ProjectionX(const char *name="_px", Option_t *option="e");
         TProfile  *Rebin(Int_t ngroup=2, const char*newname="");
    virtual void    Reset(Option_t *option="");
    virtual void    Scale(Double_t c1=1);
    virtual void    SetBinEntries(Int_t bin, Stat_t w);
    virtual void    SetBins(Int_t nbins, Double_t xmin, Double_t xmax);
    virtual void    SetErrorOption(Option_t *option=""); // *MENU*

    ClassDef(TProfile,2)  //Profile histogram class
};

#endif

