// @(#)root/hist:$Name$:$Id$
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

#ifndef ROOT_TH2
#include "TH2.h"
#endif
#ifndef ROOT_TProfile
#include "TProfile.h"
#endif

class TProfile2D : public TH2D {

protected:
    TArrayD     fBinEntries;      //number of entries per bin
    EErrorType  fErrorMode;       //Option to compute errors
    Float_t     fZmin;            //Lower limit in Z (if set)
    Float_t     fZmax;            //Upper limit in Z (if set)

private:
   Double_t *GetB()  {return &fBinEntries.fArray[0];}
   Double_t *GetW()  {return &fArray[0];}
   Double_t *GetW2() {return &fSumw2.fArray[0];}

public:
    TProfile2D();
    TProfile2D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                ,Int_t nbinsy,Axis_t ylow,Axis_t yup,Option_t *option="");
    TProfile2D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup
                                ,Int_t nbinsy,Axis_t ylow,Axis_t yup
                                ,Axis_t zlow, Axis_t zup,Option_t *option="");
    TProfile2D(const TProfile2D &profile);
    virtual ~TProfile2D();
    virtual void    Add(TH1 *h1, Float_t c1=1);
    virtual void    Add(TH1 *h1, TH1 *h2, Float_t c1=1, Float_t c2=1); // *MENU*
            void    BuildOptions(Float_t zmin, Float_t zmax, Option_t *option);
    virtual void    Copy(TObject &hnew);
    virtual void    Divide(TH1 *h1);
    virtual void    Divide(TH1 *h1, TH1 *h2, Float_t c1=1, Float_t c2=1, Option_t *option=""); // *MENU*
    virtual TH1    *DrawCopy(Option_t *option="");
            Int_t   Fill(Axis_t) {return -1;} //MayNotUse
            Int_t   Fill(Axis_t, Axis_t) {return -1; } //MayNotUse
            Int_t   Fill(Axis_t, Stat_t) {return -1; } //MayNotUse
    virtual Int_t   Fill(Axis_t x, Axis_t y, Axis_t z);
            Int_t   Fill(Axis_t x, Axis_t y, Stat_t z) {return Fill(x,y,(Axis_t)z);}
    virtual Int_t   Fill(Axis_t x, Axis_t y, Axis_t z, Stat_t w);
    virtual Stat_t  GetBinContent(Int_t bin);
    virtual Stat_t  GetBinError(Int_t bin);
    virtual Stat_t  GetBinEntries(Int_t bin);
    Option_t       *GetErrorOption() const;
    virtual Float_t GetZmin() {return fZmin;}
    virtual Float_t GetZmax() {return fZmax;}
    virtual void    Multiply(TH1 *h1);
    virtual void    Multiply(TH1 *h1, TH1 *h2, Float_t c1=1, Float_t c2=1, Option_t *option=""); // *MENU*
            TH2D   *ProjectionXY(const char *name="_pxy", Option_t *option="e");
    virtual void    Reset(Option_t *option="");
    virtual void    Scale(Float_t c1=1);
    virtual void    SetBinEntries(Int_t bin, Stat_t w);
            void    SetBins(Int_t, Float_t, Float_t)
                       { MayNotUse("SetBins(Int_t, Float_t, Float_t"); }
    virtual void    SetBins(Int_t nbinsx, Float_t xmin, Float_t xmax, Int_t nbinsy, Float_t ymin, Float_t ymax);
            void    SetBins(Int_t, Float_t, Float_t, Int_t, Float_t, Float_t, Int_t, Float_t, Float_t)
                       { MayNotUse("SetBins(Int_t, Float_t, Float_t, Int_t, Float_t, Float_t, Int_t, Float_t, Float_t"); }
    virtual void    SetErrorOption(Option_t *option=""); // *MENU*

    ClassDef(TProfile2D,1)  //Profile2D histogram class
};

#endif
