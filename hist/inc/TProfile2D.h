// @(#)root/hist:$Name:  $:$Id: TProfile2D.h,v 1.12 2002/01/20 10:21:46 brun Exp $
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
    Double_t    fZmin;            //Lower limit in Z (if set)
    Double_t    fZmax;            //Upper limit in Z (if set)

   virtual Int_t    BufferFill(Axis_t x, Stat_t w) {return -2;} //may not use
   virtual Int_t    BufferFill(Axis_t x, Axis_t y, Stat_t w) {return -2;} //may not use
   virtual Int_t    BufferFill(Axis_t x, Axis_t y, Axis_t z, Stat_t w);

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
    virtual void    Add(TF1 *h1, Double_t c1=1);
    virtual void    Add(const TH1 *h1, Double_t c1=1);
    virtual void    Add(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1); // *MENU*
            void    BuildOptions(Double_t zmin, Double_t zmax, Option_t *option);
    virtual Int_t   BufferEmpty(Bool_t deleteBuffer=kFALSE);
    virtual void    Copy(TObject &hnew);
    virtual void    Divide(TF1 *h1, Double_t c1=1);
    virtual void    Divide(const TH1 *h1);
    virtual void    Divide(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
    virtual TH1    *DrawCopy(Option_t *option="");
            Int_t   Fill(Axis_t) {return -1;} //MayNotUse
            Int_t   Fill(const char*, Stat_t) {return -1;} //MayNotUse
            Int_t   Fill(Axis_t, Stat_t) {return -1; } //MayNotUse
            Int_t   Fill(Axis_t x, Axis_t y, Axis_t z);
    virtual Int_t   Fill(Axis_t x, const char *namey, Axis_t z);
    virtual Int_t   Fill(const char *namex, Axis_t y, Axis_t z);
    virtual Int_t   Fill(const char *namex, const char *namey, Axis_t z);
    virtual Int_t   Fill(Axis_t x, Axis_t y, Axis_t z, Stat_t w);
    virtual Stat_t  GetBinContent(Int_t bin) const;
    virtual Stat_t  GetBinContent(Int_t binx, Int_t biny) const {return GetBinContent(GetBin(binx,biny));}
    virtual Stat_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const {return GetBinContent(GetBin(binx,biny));}
    virtual Stat_t  GetBinError(Int_t bin) const;
    virtual Stat_t  GetBinError(Int_t binx, Int_t biny) const {return GetBinError(GetBin(binx,biny));}
    virtual Stat_t  GetBinError(Int_t binx, Int_t biny, Int_t) const {return GetBinError(GetBin(binx,biny));}
    virtual Stat_t  GetBinEntries(Int_t bin) const;
    Option_t       *GetErrorOption() const;
    virtual void    GetStats(Stat_t *stats) const;
    virtual Double_t GetZmin() const {return fZmin;}
    virtual Double_t GetZmax() const {return fZmax;}
    virtual void    LabelsDeflate(Option_t *axis="X");
    virtual void    LabelsInflate(Option_t *axis="X");
    virtual void    LabelsOption(Option_t *option="h", Option_t *axis="X");
    virtual Int_t   Merge(TCollection *list);
    virtual void    Multiply(TF1 *h1, Double_t c1=1);
    virtual void    Multiply(const TH1 *h1);
    virtual void    Multiply(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
            TH2D   *ProjectionXY(const char *name="_pxy", Option_t *option="e") const;
    virtual void    Reset(Option_t *option="");
    virtual void    Scale(Double_t c1=1);
    virtual void    SetBinEntries(Int_t bin, Stat_t w);
            void    SetBins(Int_t, Double_t, Double_t)
                       { MayNotUse("SetBins(Int_t, Double_t, Double_t"); }
    virtual void    SetBins(Int_t nbinsx, Double_t xmin, Double_t xmax, Int_t nbinsy, Double_t ymin, Double_t ymax);
            void    SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t)
                       { MayNotUse("SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t"); }
    virtual void    SetBuffer(Int_t buffersize, Option_t *option="");
    virtual void    SetErrorOption(Option_t *option=""); // *MENU*

    ClassDef(TProfile2D,3)  //Profile2D histogram class
};

#endif
