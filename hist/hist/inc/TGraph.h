// @(#)root/hist:$Id$
// Author: Rene Brun, Olivier Couet   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraph
#define ROOT_TGraph


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraph                                                               //
//                                                                      //
// Graph graphics class.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TVectorFfwd.h"
#include "TVectorDfwd.h"
#include "TFitResultPtr.h"

class TBrowser;
class TAxis;
class TH1;
class TH1F;
class TCollection;
class TF1;
class TSpline;
class TList;

class TGraph : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:

   Int_t              fMaxSize;   ///<!Current dimension of arrays fX and fY
   Int_t              fNpoints;   ///< Number of points <= fMaxSize
   Double_t          *fX;         ///<[fNpoints] array of X points
   Double_t          *fY;         ///<[fNpoints] array of Y points
   TList             *fFunctions; ///< Pointer to list of functions (fits and user)
   TH1F              *fHistogram; ///< Pointer to histogram used for drawing axis
   Double_t           fMinimum;   ///< Minimum value for plotting along y
   Double_t           fMaximum;   ///< Maximum value for plotting along y

   static void        SwapValues(Double_t* arr, Int_t pos1, Int_t pos2);
   virtual void       SwapPoints(Int_t pos1, Int_t pos2);

   virtual Double_t **Allocate(Int_t newsize);
   Double_t         **AllocateArrays(Int_t Narrays, Int_t arraySize);
   virtual Bool_t     CopyPoints(Double_t **newarrays, Int_t ibegin, Int_t iend, Int_t obegin);
   virtual void       CopyAndRelease(Double_t **newarrays, Int_t ibegin, Int_t iend, Int_t obegin);
   Bool_t             CtorAllocate();
   Double_t         **ExpandAndCopy(Int_t size, Int_t iend);
   virtual void       FillZero(Int_t begin, Int_t end, Bool_t from_ctor = kTRUE);
   Double_t         **ShrinkAndCopy(Int_t size, Int_t iend);
   virtual Bool_t     DoMerge(const TGraph * g);

   TString            SaveArray(std::ostream &out, const char *suffix, Int_t frameNumber, Double_t *arr);
   void               SaveHistogramAndFunctions(std::ostream &out, const char *varname, Int_t &frameNumber, Option_t *option);

public:
   // TGraph status bits
   enum EStatusBits {
      kNoStats       = BIT(9),   ///< Don't draw stats box
      kClipFrame     = BIT(10),  ///< Clip to the frame boundary
      kResetHisto    = BIT(17),  ///< fHistogram must be reset in GetHistogram
      kNotEditable   = BIT(18),  ///< Bit set if graph is non editable
      kIsSortedX     = BIT(19),  ///< Graph is sorted in X points
      kIsHighlight   = BIT(20)   ///< Bit set if graph is highlight
   };

   TGraph();
   TGraph(Int_t n);
   TGraph(Int_t n, const Int_t *x, const Int_t *y);
   TGraph(Int_t n, const Float_t *x, const Float_t *y);
   TGraph(Int_t n, const Double_t *x, const Double_t *y);
   TGraph(Int_t n, const Double_t *y, Double_t start=0., Double_t step=1.);
   TGraph(const TGraph &gr);
   TGraph& operator=(const TGraph&);
   TGraph(const TVectorF &vx, const TVectorF &vy);
   TGraph(const TVectorD &vx, const TVectorD &vy);
   TGraph(const TH1 *h);
   TGraph(const TF1 *f, Option_t *option="");
   TGraph(const char *filename, const char *format="%lg %lg", Option_t *option="");
   ~TGraph() override;

   virtual void          AddPoint(Double_t x, Double_t y) { SetPoint(fNpoints, x, y); } ///< Append a new point to the graph.
   virtual void          Apply(TF1 *f);
   void          Browse(TBrowser *b) override;
   virtual Double_t      Chisquare(TF1 *f1, Option_t *option="") const;
   static Bool_t         CompareArg(const TGraph* gr, Int_t left, Int_t right);
   static Bool_t         CompareX(const TGraph* gr, Int_t left, Int_t right);
   static Bool_t         CompareY(const TGraph* gr, Int_t left, Int_t right);
   static Bool_t         CompareRadius(const TGraph* gr, Int_t left, Int_t right);
   virtual void          ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax) const;
   Int_t         DistancetoPrimitive(Int_t px, Int_t py) override;
   void          Draw(Option_t *chopt="") override;
   virtual void          DrawGraph(Int_t n, const Int_t *x, const Int_t *y, Option_t *option="");
   virtual void          DrawGraph(Int_t n, const Float_t *x, const Float_t *y, Option_t *option="");
   virtual void          DrawGraph(Int_t n, const Double_t *x=nullptr, const Double_t *y=nullptr, Option_t *option="");
   virtual void          DrawPanel(); // *MENU*
   virtual Double_t      Eval(Double_t x, TSpline *spline=nullptr, Option_t *option="") const;
   void          ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual void          Expand(Int_t newsize);
   virtual void          Expand(Int_t newsize, Int_t step);
   TObject      *FindObject(const char *name) const override;
   TObject      *FindObject(const TObject *obj) const override;
   virtual TFitResultPtr Fit(const char *formula ,Option_t *option="" ,Option_t *goption="", Axis_t xmin=0, Axis_t xmax=0); // *MENU*
   virtual TFitResultPtr Fit(TF1 *f1 ,Option_t *option="" ,Option_t *goption="", Axis_t xmin=0, Axis_t xmax=0);
   virtual void          FitPanel(); // *MENU*
   Bool_t                GetEditable() const;
   TF1                  *GetFunction(const char *name) const;
   TH1F                 *GetHistogram() const;
   TList                *GetListOfFunctions() const { return fFunctions; }
   virtual Double_t      GetCorrelationFactor() const;
   virtual Double_t      GetCovariance() const;
   virtual Double_t      GetMean(Int_t axis=1) const;
   virtual Double_t      GetRMS(Int_t axis=1) const;
   Int_t                 GetMaxSize() const {return fMaxSize;}
   Int_t                 GetN() const {return fNpoints;}
   virtual Double_t      GetErrorX(Int_t bin) const;
   virtual Double_t      GetErrorY(Int_t bin) const;
   virtual Double_t      GetErrorXhigh(Int_t bin) const;
   virtual Double_t      GetErrorXlow(Int_t bin)  const;
   virtual Double_t      GetErrorYhigh(Int_t bin) const;
   virtual Double_t      GetErrorYlow(Int_t bin)  const;
   Double_t             *GetX()  const {return fX;}
   Double_t             *GetY()  const {return fY;}
   virtual Double_t     *GetEX() const {return nullptr;}
   virtual Double_t     *GetEY() const {return nullptr;}
   virtual Double_t     *GetEXhigh() const {return nullptr;}
   virtual Double_t     *GetEXlow()  const {return nullptr;}
   virtual Double_t     *GetEYhigh() const {return nullptr;}
   virtual Double_t     *GetEYlow()  const {return nullptr;}
   virtual Double_t     *GetEXlowd()  const {return nullptr;}
   virtual Double_t     *GetEXhighd() const {return nullptr;}
   virtual Double_t     *GetEYlowd()  const {return nullptr;}
   virtual Double_t     *GetEYhighd() const {return nullptr;}
   Double_t              GetMaximum()  const {return fMaximum;}
   Double_t              GetMinimum()  const {return fMinimum;}
   TAxis                *GetXaxis() const ;
   TAxis                *GetYaxis() const ;
   char         *GetObjectInfo(Int_t px, Int_t py) const override;
   virtual Int_t         GetPoint(Int_t i, Double_t &x, Double_t &y) const;
   virtual Double_t      GetPointX(Int_t i) const;
   virtual Double_t      GetPointY(Int_t i) const;

   virtual void          InitExpo(Double_t xmin=0, Double_t xmax=0);
   virtual void          InitGaus(Double_t xmin=0, Double_t xmax=0);
   virtual void          InitPolynom(Double_t xmin=0, Double_t xmax=0);
   virtual Int_t         InsertPoint(); // *MENU*
   virtual void          InsertPointBefore(Int_t ipoint, Double_t x, Double_t y);
   virtual Double_t      Integral(Int_t first=0, Int_t last=-1) const;
   virtual Bool_t        IsEditable() const {return !TestBit(kNotEditable);}
   virtual Bool_t        IsHighlight() const { return TestBit(kIsHighlight); }
   virtual Int_t         IsInside(Double_t x, Double_t y) const;
   virtual void          LeastSquareFit(Int_t m, Double_t *a, Double_t xmin=0, Double_t xmax=0);
   virtual void          LeastSquareLinearFit(Int_t n, Double_t &a0, Double_t &a1, Int_t &ifail, Double_t xmin=0, Double_t xmax=0);
   virtual Int_t         Merge(TCollection* list);
   virtual void          MovePoints(Double_t dx, Double_t dy, Bool_t logx = kFALSE, Bool_t logy = kFALSE);
   void          Paint(Option_t *chopt="") override;
   void                  PaintGraph(Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt);
   void                  PaintGrapHist(Int_t npoints, const Double_t *x, const Double_t *y, Option_t *chopt);
   virtual void          PaintStats(TF1 *fit);
   void          Print(Option_t *chopt="") const override;
   void          RecursiveRemove(TObject *obj) override;
   virtual Int_t         RemovePoint(); // *MENU*
   virtual Int_t         RemovePoint(Int_t ipoint);
   void          SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void          SaveAs(const char *filename, Option_t *option = "") const override; // *MENU*
   virtual void          Scale(Double_t c1=1., Option_t *option="y"); // *MENU*
   virtual void          SetEditable(Bool_t editable=kTRUE); // *TOGGLE* *GETTER=GetEditable
   virtual void          SetHighlight(Bool_t set = kTRUE); // *TOGGLE* *GETTER=IsHighlight
   virtual void          SetHistogram(TH1F *h) {fHistogram = h;}
   virtual void          SetMaximum(Double_t maximum=-1111); // *MENU*
   virtual void          SetMinimum(Double_t minimum=-1111); // *MENU*
   virtual void          Set(Int_t n);
   virtual void          SetPoint(Int_t i, Double_t x, Double_t y);
   virtual void          SetPointX(Int_t i, Double_t x);
   virtual void          SetPointY(Int_t i, Double_t y);
   void          SetName(const char *name="") override; // *MENU*
   void          SetNameTitle(const char *name="", const char *title="") override;
   virtual void          SetStats(Bool_t stats=kTRUE); // *MENU*
   void          SetTitle(const char *title="") override;    // *MENU*
   virtual void          Sort(Bool_t (*greater)(const TGraph*, Int_t, Int_t)=&TGraph::CompareX,
                              Bool_t ascending=kTRUE, Int_t low=0, Int_t high=-1111);
   void          UseCurrentStyle() override;
   void                  Zero(Int_t &k,Double_t AZ,Double_t BZ,Double_t E2,Double_t &X,Double_t &Y,Int_t maxiterations);

   ClassDefOverride(TGraph,4)  //Graph graphics class
};

#endif
