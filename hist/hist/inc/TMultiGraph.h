// @(#)root/hist:$Id$
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMultiGraph
#define ROOT_TMultiGraph


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMultiGraph                                                          //
//                                                                      //
// A collection of TGraph objects                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TCollection.h"
#include "TFitResultPtr.h"

class TH1F;
class TAxis;
class TBrowser;
class TGraph;
class TF1;

class TMultiGraph : public TNamed {

protected:
   TList      *fGraphs{nullptr};     ///< Pointer to list of TGraphs
   TList      *fFunctions{nullptr};  ///< Pointer to list of functions (fits and user)
   TH1F       *fHistogram{nullptr};  ///< Pointer to histogram used for drawing axis
   Double_t    fMaximum{-1111};      ///< Maximum value for plotting along y
   Double_t    fMinimum{-1111};      ///< Minimum value for plotting along y

   TMultiGraph(const TMultiGraph&) = delete;
   TMultiGraph& operator=(const TMultiGraph&) = delete;

public:
   TMultiGraph();
   TMultiGraph(const char *name, const char *title);
   ~TMultiGraph() override;

   virtual void      Add(TGraph *graph, Option_t *chopt = "");
   virtual void      Add(TMultiGraph *multigraph, Option_t *chopt = "");
   void              Browse(TBrowser *b) override;
   Int_t             DistancetoPrimitive(Int_t px, Int_t py) override;
   void              Draw(Option_t *chopt = "") override;
   virtual TFitResultPtr Fit(const char *formula ,Option_t *option="" ,Option_t *goption="", Axis_t xmin=0, Axis_t xmax=0);
   virtual TFitResultPtr Fit(TF1 *f1 ,Option_t *option="" ,Option_t *goption="", Axis_t rxmin=0, Axis_t rxmax=0);
   virtual void      FitPanel(); // *MENU*
   virtual Option_t *GetGraphDrawOption(const TGraph *gr) const;
   virtual void      LeastSquareLinearFit(Int_t ndata, Double_t &a0, Double_t &a1, Int_t &ifail, Double_t xmin, Double_t xmax);
   virtual void      LeastSquareFit(Int_t m, Double_t *a, Double_t xmin, Double_t xmax);
   virtual void      InitPolynom(Double_t xmin, Double_t xmax);
   virtual void      InitExpo(Double_t xmin, Double_t xmax);
   virtual void      InitGaus(Double_t xmin, Double_t xmax);
   virtual Int_t     IsInside(Double_t x, Double_t y) const;
   TH1F             *GetHistogram();
   TF1              *GetFunction(const char *name) const;
   TList            *GetListOfGraphs() const { return fGraphs; }
   TIter             begin() const;
   TIter             end() const { return TIter::End(); }
   TList            *GetListOfFunctions();  // non const method (create list if empty)
   const TList      *GetListOfFunctions() const { return fFunctions; }
   TAxis            *GetXaxis();
   TAxis            *GetYaxis();
   void              Paint(Option_t *chopt = "") override;
   void              PaintPads(Option_t *chopt = "");
   void              PaintPolyLine3D(Option_t *chopt = "");
   void              PaintReverse(Option_t *chopt = "");
   void              Print(Option_t *chopt="") const override;
   void              RecursiveRemove(TObject *obj) override;
   void              SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void      SetMaximum(Double_t maximum=-1111);
   virtual void      SetMinimum(Double_t minimum=-1111);

   ClassDefOverride(TMultiGraph,2)  // A collection of TGraph objects
};

#endif
