// @(#)root/graf:$Name:  $:$Id: TGraph.h,v 1.7 2000/12/08 16:00:33 brun Exp $
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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TH1
#include "TH1.h"
#endif

class TBrowser;
class TF1;

class TGraph : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:
    Int_t       fNpoints;    //Number of points
    Double_t   *fX;          //[fNpoints] array of X points
    Double_t   *fY;          //[fNpoints] array of Y points
    Option_t   *fOption;     //!Axis options
    TList      *fFunctions;  //Pointer to list of functions (fits and user)
    TH1F       *fHistogram;  //Pointer to histogram used for drawing axis
    Double_t    fMinimum;    //Minimum value for plotting along y
    Double_t    fMaximum;    //Maximum value for plotting along y

    virtual void    LeastSquareFit(Int_t n, Int_t m, Double_t *a);
    virtual void    LeastSquareLinearFit(Int_t ndata, Double_t &a0, Double_t &a1, Int_t &ifail);

public:
    // TGraph status bits
    enum {
       kClipFrame     = BIT(10),  // clip to the frame boundary
       kFitInit       = BIT(19)
    };

        TGraph();
        TGraph(Int_t n);
        TGraph(Int_t n, Float_t *x, Float_t *y);
        TGraph(Int_t n, Double_t *x, Double_t *y);
        virtual ~TGraph();
        virtual void     Browse(TBrowser *b);
                void     ComputeLogs(Int_t npoints, Int_t opt);
        virtual void     ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual void     Draw(Option_t *chopt="");
        virtual void     DrawGraph(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void     DrawGraph(Int_t n, Double_t *x, Double_t *y, Option_t *option="");
        virtual void     DrawPanel(); // *MENU*
        virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual void     Fit(const char *formula ,Option_t *option="" ,Option_t *goption=""); // *MENU*
        virtual void     Fit(TF1 *f1 ,Option_t *option="" ,Option_t *goption=""); // *MENU*
        virtual void     FitPanel(); // *MENU*
        virtual Double_t GetErrorX(Int_t bin) const;
        virtual Double_t GetErrorY(Int_t bin) const;
        TF1             *GetFunction(const char *name) const;
        TH1F            *GetHistogram() const;
        TList           *GetListOfFunctions() const { return fFunctions; }
        Int_t            GetN() const {return fNpoints;}
        Double_t        *GetX() const {return fX;}
        Double_t        *GetY() const {return fY;}
        TAxis           *GetXaxis() const ;
        TAxis           *GetYaxis() const ;
        virtual void     GetPoint(Int_t i, Double_t &x, Double_t &y);
        virtual void     InitExpo();
        virtual void     InitGaus();
        virtual void     InitPolynom();
        virtual void     Paint(Option_t *chopt="");
        virtual void     PaintGraph(Int_t npoints, Double_t *x, Double_t *y, Option_t *option="");
        virtual void     PaintGrapHist(Int_t npoints, Double_t *x, Double_t *y, Option_t *option="");
        virtual void     Print(Option_t *chopt="") const;
        static  void     RemoveFunction(TGraph *gr, TObject *obj);
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetMaximum(Double_t maximum=-1111); // *MENU*
        virtual void     SetMinimum(Double_t minimum=-1111); // *MENU*
        virtual void     Set(Int_t n);
        virtual void     SetPoint(Int_t i, Double_t x, Double_t y);
        virtual void     SetTitle(const char *title="");    // *MENU*
                void     Smooth(Int_t npoints, Double_t *x, Double_t *y, Int_t drawtype);
        virtual void     UseCurrentStyle();
                void     Zero(Int_t &k,Double_t AZ,Double_t BZ,Double_t E2,Double_t &X,Double_t &Y
                          ,Int_t maxiterations);

        ClassDef(TGraph,2)  //Graph graphics class
};

#endif


