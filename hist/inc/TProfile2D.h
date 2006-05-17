// @(#)root/hist:$Name:  $:$Id: TProfile2D.h,v 1.27 2005/12/04 10:51:27 brun Exp $
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
    Bool_t      fScaling;         //!True when TProfile2D::Scale is called
    Double_t    fTsumwz;          //Total Sum of weight*Z
    Double_t    fTsumwz2;         //Total Sum of weight*Z*Z
static Bool_t   fgApproximate;    //bin error approximation option

   virtual Int_t    BufferFill(Double_t, Double_t) {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t, Double_t, Double_t) {return -2;} //may not use
   virtual Int_t    BufferFill(Double_t x, Double_t y, Double_t z, Double_t w);

private:
   Double_t *GetB()  {return &fBinEntries.fArray[0];}
   Double_t *GetW()  {return &fArray[0];}
   Double_t *GetW2() {return &fSumw2.fArray[0];}

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
   virtual ~TProfile2D();
   virtual void      Add(TF1 *h1, Double_t c1=1, Option_t *option="");
   virtual void      Add(const TH1 *h1, Double_t c1=1);
   virtual void      Add(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1); // *MENU*
   static  void      Approximate(Bool_t approx=kTRUE);
   void              BuildOptions(Double_t zmin, Double_t zmax, Option_t *option);
   virtual Int_t     BufferEmpty(Int_t action=0);
   virtual void      Copy(TObject &hnew) const;
   virtual void      Divide(TF1 *h1, Double_t c1=1);
   virtual void      Divide(const TH1 *h1);
   virtual void      Divide(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
   virtual TH1      *DrawCopy(Option_t *option="") const;
   Int_t             Fill(Double_t) {return -1;} //MayNotUse
   Int_t             Fill(const char*, Double_t) {return -1;} //MayNotUse
   Int_t             Fill(Double_t, Double_t) {return -1; } //MayNotUse
   Int_t             Fill(Double_t x, Double_t y, Double_t z);
   virtual Int_t     Fill(Double_t x, const char *namey, Double_t z);
   virtual Int_t     Fill(const char *namex, Double_t y, Double_t z);
   virtual Int_t     Fill(const char *namex, const char *namey, Double_t z);
   virtual Int_t     Fill(Double_t x, Double_t y, Double_t z, Double_t w);
   virtual Double_t  GetBinContent(Int_t bin) const;
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny) const {return GetBinContent(GetBin(binx,biny));}
   virtual Double_t  GetBinContent(Int_t binx, Int_t biny, Int_t) const {return GetBinContent(GetBin(binx,biny));}
   virtual Double_t  GetBinError(Int_t bin) const;
   virtual Double_t  GetBinError(Int_t binx, Int_t biny) const {return GetBinError(GetBin(binx,biny));}
   virtual Double_t  GetBinError(Int_t binx, Int_t biny, Int_t) const {return GetBinError(GetBin(binx,biny));}
   virtual Double_t  GetBinEntries(Int_t bin) const;
   Option_t         *GetErrorOption() const;
   virtual void      GetStats(Double_t *stats) const;
   virtual Double_t  GetZmin() const {return fZmin;}
   virtual Double_t  GetZmax() const {return fZmax;}
   virtual void      LabelsDeflate(Option_t *axis="X");
   virtual void      LabelsInflate(Option_t *axis="X");
   virtual void      LabelsOption(Option_t *option="h", Option_t *axis="X");
   virtual Long64_t  Merge(TCollection *list);
   virtual void      Multiply(TF1 *h1, Double_t c1=1);
   virtual void      Multiply(const TH1 *h1);
   virtual void      Multiply(const TH1 *h1, const TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
   TH2D             *ProjectionXY(const char *name="_pxy", Option_t *option="e") const;
   virtual void      PutStats(Double_t *stats);
   virtual void      Reset(Option_t *option="");
   virtual void      RebinAxis(Double_t x, const char *ax);
   virtual void      SavePrimitive(ofstream &out, Option_t *option);
   virtual void      Scale(Double_t c1=1);
   virtual void      SetBinEntries(Int_t bin, Double_t w);
   void              SetBins(Int_t, Double_t, Double_t)
                      { MayNotUse("SetBins(Int_t, Double_t, Double_t"); }
   void              SetBins(Int_t, const Double_t*)
                      { MayNotUse("SetBins(Int_t, const Double_t*"); }
   virtual void      SetBins(Int_t nbinsx, Double_t xmin, Double_t xmax, Int_t nbinsy, Double_t ymin, Double_t ymax);
   void              SetBins(Int_t, const Double_t*, Int_t, const Double_t*)
                      { MayNotUse("SetBins(Int_t, const Double_t*, Int_t, const Double_t*"); }
   void              SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t)
                      { MayNotUse("SetBins(Int_t, Double_t, Double_t, Int_t, Double_t, Double_t, Int_t, Double_t, Double_t"); }
   virtual void      SetBuffer(Int_t buffersize, Option_t *option="");
   virtual void      SetErrorOption(Option_t *option=""); // *MENU*

   ClassDef(TProfile2D,6)  //Profile2D histogram class
};

#endif
