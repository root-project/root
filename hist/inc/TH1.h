// @(#)root/hist:$Name:  $:$Id: TH1.h,v 1.24 2001/12/10 13:50:50 rdm Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TH1
#define ROOT_TH1


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TH1                                                                  //
//                                                                      //
// 1-Dim histogram base class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualHistPainter
#include "TVirtualHistPainter.h"
#endif

#ifndef ROOT_TAxis
#include "TAxis.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif

#ifndef ROOT_TArrayC
#include "TArrayC.h"
#endif
#ifndef ROOT_TArrayS
#include "TArrayS.h"
#endif
#ifndef ROOT_TArrayF
#include "TArrayF.h"
#endif
#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif

class TF1;
class TH1D;
class TBrowser;
class TDirectory;
class TVector;
class TVectorD;

class TH1 : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:
    Int_t         fNcells;          //number of bins(1D), cells (2D) +U/Overflows
    TAxis         fXaxis;           //X axis descriptor
    TAxis         fYaxis;           //Y axis descriptor
    TAxis         fZaxis;           //Z axis descriptor
    Short_t       fBarOffset;       //(1000*offset) for bar charts or legos
    Short_t       fBarWidth;        //(1000*width) for bar charts or legos
    Stat_t        fEntries;         //Number of entries
    Stat_t        fTsumw;           //Total Sum of weights
    Stat_t        fTsumw2;          //Total Sum of squares of weights
    Stat_t        fTsumwx;          //Total Sum of weight*X
    Stat_t        fTsumwx2;         //Total Sum of weight*X*X
    Double_t      fMaximum;         //Maximum value for plotting
    Double_t      fMinimum;         //Minimum value for plotting
    Double_t      fNormFactor;      //Normalization factor
    TArrayD       fContour;         //Array to display contour levels
    TArrayD       fSumw2;           //Array of sum of squares of weights
    TString       fOption;          //histogram options
    TList        *fFunctions;       //->Pointer to list of functions (fits and user)
    TDirectory   *fDirectory;       //!Pointer to directory holding this histogram
    Int_t         fDimension;       //!Histogram dimension (1, 2 or 3 dim)
    Double_t     *fIntegral;        //!Integral of bins used by GetRandom
    TVirtualHistPainter *fPainter;  //!pointer to histogram painter
    static Bool_t fgAddDirectory;   //!flag to add histograms to the directory
private:
    Int_t   AxisChoice(Option_t *axis) const;
    void    Build();
    Int_t   FitOptionsMake(Option_t *option);

protected:
    virtual void    Copy(TObject &hnew);

public:
    // TH1 status bits
    enum {
       kNoStats     = BIT(9),  // don't draw stats box
       kUserContour = BIT(10), // user specified contour levels
       kCanRebin    = BIT(11), // can rebin axis
       kLogX        = BIT(15)  // X-axis in log scale
    };

    TH1();
    TH1(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup);
    TH1(const char *name,const char *title,Int_t nbinsx,const Float_t *xbins);
    TH1(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
    virtual ~TH1();

    virtual void     Add(TF1 *h1, Double_t c1=1);
    virtual void     Add(TH1 *h1, Double_t c1=1);
    virtual void     Add(TH1 *h1, TH1 *h2, Double_t c1=1, Double_t c2=1); // *MENU*
    virtual void     AddBinContent(Int_t bin);
    virtual void     AddBinContent(Int_t bin, Stat_t w);
    static  void     AddDirectory(Bool_t add=kTRUE);
    static  Bool_t   AddDirectoryStatus();
    virtual void     Browse(TBrowser *b);
    virtual Double_t ComputeIntegral();
    virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
    virtual void     Divide(TF1 *f1, Double_t c1=1);
    virtual void     Divide(TH1 *h1);
    virtual void     Divide(TH1 *h1, TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
    virtual void     Draw(Option_t *option="");
    virtual TH1     *DrawCopy(Option_t *option="");
    virtual void     DrawPanel(); // *MENU*
    virtual void     Eval(TF1 *f1, Option_t *option="");
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual Int_t    Fill(Axis_t x);
    virtual Int_t    Fill(Axis_t x, Stat_t w);
    virtual void     FillN(Int_t ntimes, const Axis_t *x, const Double_t *w, Int_t stride=1);
    virtual void     FillN(Int_t, const Axis_t *, const Axis_t *, const Double_t *, Int_t) {;}
    virtual void     FillRandom(const char *fname, Int_t ntimes=5000);
    virtual void     FillRandom(TH1 *h, Int_t ntimes=5000);
    virtual Int_t    FindBin(Axis_t x, Axis_t y=0, Axis_t z=0);
    virtual void     Fit(const char *formula ,Option_t *option="" ,Option_t *goption="", Axis_t xmin=0, Axis_t xmax=0); // *MENU*
    virtual void     Fit(TF1 *f1 ,Option_t *option="" ,Option_t *goption="", Axis_t xmin=0, Axis_t xmax=0);
    virtual void     FitPanel(); // *MENU*
    virtual Double_t *GetIntegral() {return fIntegral;}

    TList           *GetListOfFunctions() const { return fFunctions; }

    virtual Int_t    GetNdivisions(Option_t *axis="X") const;
    virtual Color_t  GetAxisColor(Option_t *axis="X") const;
    virtual Color_t  GetLabelColor(Option_t *axis="X") const;
    virtual Style_t  GetLabelFont(Option_t *axis="X") const;
    virtual Float_t  GetLabelOffset(Option_t *axis="X") const;
    virtual Float_t  GetLabelSize(Option_t *axis="X") const;
    virtual Float_t  GetTitleOffset(Option_t *axis="X") const;
    virtual Float_t  GetTitleSize(Option_t *axis="X") const;
    virtual Float_t  GetTickLength(Option_t *axis="X") const;
    virtual Float_t  GetBarOffset() const {return Float_t(0.001*Float_t(fBarOffset));}
    virtual Float_t  GetBarWidth() const  {return Float_t(0.001*Float_t(fBarWidth));}
    virtual Int_t    GetContour(Double_t *levels=0);
    virtual Double_t GetContourLevel(Int_t level) const;

    virtual Int_t    GetBin(Int_t binx, Int_t biny=0, Int_t binz=0) const;
    virtual Axis_t   GetBinCenter(Int_t bin) const {return fXaxis.GetBinCenter(bin);}
    virtual Stat_t   GetBinContent(Int_t bin) const;
    virtual Stat_t   GetBinContent(Int_t binx, Int_t biny) const;
    virtual Stat_t   GetBinContent(Int_t binx, Int_t biny, Int_t binz) const;
    virtual Stat_t   GetBinError(Int_t bin) const;
    virtual Stat_t   GetBinError(Int_t binx, Int_t biny) const;
    virtual Stat_t   GetBinError(Int_t binx, Int_t biny, Int_t binz) const;
    virtual Axis_t   GetBinLowEdge(Int_t bin) const {return fXaxis.GetBinLowEdge(bin);}
    virtual Axis_t   GetBinWidth(Int_t bin) const {return fXaxis.GetBinWidth(bin);}
    virtual Stat_t   GetCellContent(Int_t binx, Int_t biny) const;
    virtual Stat_t   GetCellError(Int_t binx, Int_t biny) const;
    virtual void     GetCenter(Axis_t *center) {fXaxis.GetCenter(center);}
    TDirectory      *GetDirectory() const {return fDirectory;}
    virtual Stat_t   GetEntries() const {return fEntries;}
    virtual TF1     *GetFunction(const char *name) const;
    virtual Int_t    GetDimension() const { return fDimension; }
    virtual void     GetLowEdge(Axis_t *edge) {fXaxis.GetLowEdge(edge);}
    virtual Double_t GetMaximum() const;
    virtual Int_t    GetMaximumBin() const;
    virtual Int_t    GetMaximumBin(Int_t &locmax, Int_t &locmay, Int_t &locmaz) const;
    virtual Double_t GetMaximumStored() const {return fMaximum;}
    virtual Double_t GetMinimum() const;
    virtual Int_t    GetMinimumBin() const;
    virtual Int_t    GetMinimumBin(Int_t &locmix, Int_t &locmiy, Int_t &locmiz) const;
    virtual Double_t GetMinimumStored() const {return fMinimum;}
    virtual Stat_t   GetMean(Int_t axis=1) const;
    virtual Int_t    GetNbinsX() const {return fXaxis.GetNbins();}
    virtual Int_t    GetNbinsY() const {return fYaxis.GetNbins();}
    virtual Int_t    GetNbinsZ() const {return fZaxis.GetNbins();}
    virtual Double_t GetNormFactor() const {return fNormFactor;}
    virtual char    *GetObjectInfo(Int_t px, Int_t py) const;
    Option_t        *GetOption() const {return fOption.Data();}

    TVirtualHistPainter *GetPainter();

    virtual Int_t    GetQuantiles(Int_t nprobSum, Double_t *q, const Double_t *probSum=0);
    virtual Axis_t   GetRandom();
    virtual void     GetStats(Stat_t *stats) const;
    virtual Stat_t   GetSumOfWeights() const;
    virtual Int_t    GetSumw2N() const {return fSumw2.fN;}
    virtual Stat_t   GetRMS(Int_t axis=1) const;
    virtual TAxis   *GetXaxis() {return &fXaxis;}
    virtual TAxis   *GetYaxis() {return &fYaxis;}
    virtual TAxis   *GetZaxis() {return &fZaxis;}
    virtual Stat_t   Integral(Option_t *option="");
    virtual Stat_t   Integral(Int_t binx1, Int_t binx2, Option_t *option="");
    virtual Stat_t   Integral(Int_t, Int_t, Int_t, Int_t, Option_t * /*option*/ ="") {return 0;}
    virtual Stat_t   Integral(Int_t, Int_t, Int_t, Int_t, Int_t, Int_t, Option_t * /*option*/ ="" ) {return 0;}
    virtual Double_t KolmogorovTest(TH1 *h2, Option_t *option="");
    virtual void     Multiply(TF1 *h1, Double_t c1=1);
    virtual void     Multiply(TH1 *h1);
    virtual void     Multiply(TH1 *h1, TH1 *h2, Double_t c1=1, Double_t c2=1, Option_t *option=""); // *MENU*
    virtual void     Paint(Option_t *option="");
    virtual void     Print(Option_t *option="") const;
    virtual void     PutStats(Stat_t *stats);
    virtual TH1     *Rebin(Int_t ngroup=2, const char*newname="");  // *MENU*
    virtual void     RebinAxis(Axis_t x, Option_t *axis="X");
    virtual void     Reset(Option_t *option="");
    virtual void     SavePrimitive(ofstream &out, Option_t *option);
    virtual void     Scale(Double_t c1=1);
    virtual void     SetAxisColor(Color_t color=1, Option_t *axis="X");
    virtual void     SetAxisRange(Axis_t xmin, Axis_t xmax, Option_t *axis="X");
    virtual void     SetBarOffset(Float_t offset=0.25) {fBarOffset = Short_t(1000*offset);}
    virtual void     SetBarWidth(Float_t width=0.5) {fBarWidth = Short_t(1000*width);}
    virtual void     SetBinContent(Int_t bin, Stat_t content);
    virtual void     SetBinContent(Int_t binx, Int_t biny, Stat_t content);
    virtual void     SetBinContent(Int_t binx, Int_t biny, Int_t binz, Stat_t content);
    virtual void     SetBinError(Int_t bin, Stat_t error);
    virtual void     SetBinError(Int_t binx, Int_t biny, Stat_t error);
    virtual void     SetBinError(Int_t binx, Int_t biny, Int_t binz, Stat_t error);
    virtual void     SetBins(Int_t nx, Axis_t xmin, Axis_t xmax);
    virtual void     SetBins(Int_t nx, Axis_t xmin, Axis_t xmax, Int_t ny, Axis_t ymin, Axis_t ymax);
    virtual void     SetBins(Int_t nx, Axis_t xmin, Axis_t xmax, Int_t ny, Axis_t ymin, Axis_t ymax,
                             Int_t nz, Axis_t zmin, Axis_t zmax);
    virtual void     SetBinsLength(Int_t) {;} //refefined in derived classes
    virtual void     SetCellContent(Int_t binx, Int_t biny, Stat_t content);
    virtual void     SetCellError(Int_t binx, Int_t biny, Stat_t content);
    virtual void     SetContent(const Stat_t *content);
    virtual void     SetContour(Int_t nlevels, const Double_t *levels=0);
    virtual void     SetContourLevel(Int_t level, Double_t value);
    virtual void     SetDirectory(TDirectory *dir);
    virtual void     SetEntries(Stat_t n) {fEntries = n;};
    virtual void     SetError(const Stat_t *error);
    virtual void     SetLabelColor(Color_t color=1, Option_t *axis="X");
    virtual void     SetLabelFont(Style_t font=62, Option_t *axis="X");
    virtual void     SetLabelOffset(Float_t offset=0.005, Option_t *axis="X");
    virtual void     SetLabelSize(Float_t size=0.02, Option_t *axis="X");

    virtual void     SetMaximum(Double_t maximum=-1111); // *MENU*
    virtual void     SetMinimum(Double_t minimum=-1111); // *MENU*
    virtual void     SetName(const char *name); // *MENU*
    virtual void     SetNameTitle(const char *name, const char *title);
    virtual void     SetNdivisions(Int_t n=510, Option_t *axis="X");
    virtual void     SetNormFactor(Double_t factor=1) {fNormFactor = factor;}
    virtual void     SetStats(Bool_t stats=kTRUE);
    virtual void     SetOption(Option_t *option=" ") {fOption = option;}
    virtual void     SetTickLength(Float_t length=0.02, Option_t *axis="X");
    virtual void     SetTitleOffset(Float_t offset=1, Option_t *axis="X");
    virtual void     SetTitleSize(Float_t size=0.02, Option_t *axis="X");
    virtual void     SetXTitle(const char *title) {fXaxis.SetTitle(title);}
    virtual void     SetYTitle(const char *title) {fYaxis.SetTitle(title);}
    virtual void     SetZTitle(const char *title) {fZaxis.SetTitle(title);}
    virtual void     Smooth(Int_t ntimes=1); // *MENU*
    static  void     SmoothArray(Int_t NN, Double_t *XX, Int_t ntimes=1);
    static Double_t  SmoothMedian(Int_t n, Double_t *a);

    virtual void     Sumw2();
    void             UseCurrentStyle();

    ClassDef(TH1,3)  //1-Dim histogram base class
};

//________________________________________________________________________

class TH1C : public TH1, public TArrayC {

public:
    TH1C();
    TH1C(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup);
    TH1C(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
    TH1C(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
    TH1C(const TH1C &h1c);
    virtual ~TH1C();

    virtual void    AddBinContent(Int_t bin);
    virtual void    AddBinContent(Int_t bin, Stat_t w);
    virtual void    Copy(TObject &hnew);
    virtual TH1    *DrawCopy(Option_t *option="");
    virtual Stat_t  GetBinContent(Int_t bin) const;
    virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
    virtual Stat_t  GetBinContent(Int_t bin, Int_t, Int_t) const {return GetBinContent(bin);}
    virtual void    Reset(Option_t *option="");
    virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Char_t (content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinsLength(Int_t nx) {TArrayC::Set(nx);}
            TH1C&   operator=(const TH1C &h1);
    friend  TH1C    operator*(Double_t c1, TH1C &h1);
    friend  TH1C    operator*(TH1C &h1, Double_t c1) {return operator*(c1,h1);}
    friend  TH1C    operator+(TH1C &h1, TH1C &h2);
    friend  TH1C    operator-(TH1C &h1, TH1C &h2);
    friend  TH1C    operator*(TH1C &h1, TH1C &h2);
    friend  TH1C    operator/(TH1C &h1, TH1C &h2);

    ClassDef(TH1C,1)  //1-Dim histograms (one char per channel)
};

//________________________________________________________________________

class TH1S : public TH1, public TArrayS {

public:
    TH1S();
    TH1S(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup);
    TH1S(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
    TH1S(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
    TH1S(const TH1S &h1s);
    virtual ~TH1S();

    virtual void    AddBinContent(Int_t bin);
    virtual void    AddBinContent(Int_t bin, Stat_t w);
    virtual void    Copy(TObject &hnew);
    virtual TH1    *DrawCopy(Option_t *option="");
    virtual Stat_t  GetBinContent(Int_t bin) const;
    virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
    virtual Stat_t  GetBinContent(Int_t bin, Int_t, Int_t) const {return GetBinContent(bin);}
    virtual void    Reset(Option_t *option="");
    virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Short_t (content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinsLength(Int_t nx) {TArrayS::Set(nx);}
            TH1S&   operator=(const TH1S &h1);
    friend  TH1S    operator*(Double_t c1, TH1S &h1);
    friend  TH1S    operator*(TH1S &h1, Double_t c1) {return operator*(c1,h1);}
    friend  TH1S    operator+(TH1S &h1, TH1S &h2);
    friend  TH1S    operator-(TH1S &h1, TH1S &h2);
    friend  TH1S    operator*(TH1S &h1, TH1S &h2);
    friend  TH1S    operator/(TH1S &h1, TH1S &h2);

    ClassDef(TH1S,1)  //1-Dim histograms (one short per channel)
};

//________________________________________________________________________

class TH1F : public TH1, public TArrayF {

public:
    TH1F();
    TH1F(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup);
    TH1F(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
    TH1F(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
    TH1F(const TVector &v);
    TH1F(const TH1F &h1f);
    virtual ~TH1F();

    virtual void    AddBinContent(Int_t bin) {++fArray[bin];}
    virtual void    AddBinContent(Int_t bin, Stat_t w)
                                 {fArray[bin] += Float_t (w);}
    virtual void    Copy(TObject &hnew);
    virtual TH1    *DrawCopy(Option_t *option="");
    virtual Stat_t  GetBinContent(Int_t bin) const;
    virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
    virtual Stat_t  GetBinContent(Int_t bin, Int_t, Int_t) const {return GetBinContent(bin);}
    virtual void    Reset(Option_t *option="");
    virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Float_t (content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinsLength(Int_t nx) {TArrayF::Set(nx);}
            TH1F&   operator=(const TH1F &h1);
    friend  TH1F    operator*(Double_t c1, TH1F &h1);
    friend  TH1F    operator*(TH1F &h1, Double_t c1) {return operator*(c1,h1);}
    friend  TH1F    operator+(TH1F &h1, TH1F &h2);
    friend  TH1F    operator-(TH1F &h1, TH1F &h2);
    friend  TH1F    operator*(TH1F &h1, TH1F &h2);
    friend  TH1F    operator/(TH1F &h1, TH1F &h2);

    ClassDef(TH1F,1)  //1-Dim histograms (one float per channel)
};

//________________________________________________________________________

class TH1D : public TH1, public TArrayD {

public:
    TH1D();
    TH1D(const char *name,const char *title,Int_t nbinsx,Axis_t xlow,Axis_t xup);
    TH1D(const char *name,const char *title,Int_t nbinsx,const Float_t  *xbins);
    TH1D(const char *name,const char *title,Int_t nbinsx,const Double_t *xbins);
    TH1D(const TVectorD &v);
    TH1D(const TH1D &h1d);
    virtual ~TH1D();

    virtual void    AddBinContent(Int_t bin) {++fArray[bin];}
    virtual void    AddBinContent(Int_t bin, Stat_t w)
                                 {fArray[bin] += Double_t (w);}
    virtual void    Copy(TObject &hnew);
    virtual TH1    *DrawCopy(Option_t *option="");
    virtual Stat_t  GetBinContent(Int_t bin) const;
    virtual Stat_t  GetBinContent(Int_t bin, Int_t) const {return GetBinContent(bin);}
    virtual Stat_t  GetBinContent(Int_t bin, Int_t, Int_t) const {return GetBinContent(bin);}
    virtual void    Reset(Option_t *option="");
    virtual void    SetBinContent(Int_t bin, Stat_t content)
                                 {fArray[bin] = Double_t (content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinContent(Int_t bin, Int_t, Int_t, Stat_t content) {SetBinContent(bin,content);}
    virtual void    SetBinsLength(Int_t nx) {TArrayD::Set(nx);}
            TH1D&   operator=(const TH1D &h1);
    friend  TH1D    operator*(Double_t c1, TH1D &h1);
    friend  TH1D    operator*(TH1D &h1, Double_t c1) {return operator*(c1,h1);}
    friend  TH1D    operator+(TH1D &h1, TH1D &h2);
    friend  TH1D    operator-(TH1D &h1, TH1D &h2);
    friend  TH1D    operator*(TH1D &h1, TH1D &h2);
    friend  TH1D    operator/(TH1D &h1, TH1D &h2);

    ClassDef(TH1D,1)  //1-Dim histograms (one double per channel)
};

   extern TH1 *R__H(Int_t hid);
   extern TH1 *R__H(const char *hname);

#endif
