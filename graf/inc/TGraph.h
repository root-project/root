// @(#)root/graf:$Name$:$Id$
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
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif


class TH1F;
class TAxis;
class TBrowser;
class TF1;

class TGraph : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:
    Int_t      fNpoints;     //Number of points
    Float_t    *fX;          //[fNpoints] array of X points
    Float_t    *fY;          //[fNpoints] array of Y points
    Option_t   *fOption;     //Axis options
    TList      *fFunctions;  //Pointer to list of functions (fits and user)
    TH1F       *fHistogram;  //Pointer to histogram used for drawing axis
    Float_t    fMaximum;     //Maximum value for plotting along y
    Float_t    fMinimum;     //Minimum value for plotting along y

    virtual void    LeastSquareFit(Int_t n, Int_t m, Double_t *a);
    virtual void    LeastSquareLinearFit(Int_t ndata, Double_t &a0, Double_t &a1, Int_t &ifail);

public:
    // TGraph status bits
    enum {
       kClipFrame     = BIT(10),  // clip to the frame boundary
       kFitInit       = BIT(19)
    };

        TGraph();
        TGraph(Int_t n, Float_t *x=0, Float_t *y=0);
        TGraph(Int_t n, Double_t *x, Double_t *y);
        virtual ~TGraph();
        virtual void    Browse(TBrowser *b);
                void    ComputeLogs(Int_t npoints, Int_t opt);
        virtual void    ComputeRange(Float_t &xmin, Float_t &ymin, Float_t &xmax, Float_t &ymax);
        virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
        virtual void    Draw(Option_t *chopt="");
        virtual void    DrawGraph(Int_t n, Float_t *x, Float_t *y, Option_t *option="");
        virtual void    DrawPanel(); // *MENU*
        virtual void    ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual void    Fit(const char *formula ,Option_t *option="" ,Option_t *goption=""); // *MENU*
        virtual void    FitPanel(); // *MENU*
        virtual Float_t GetErrorX(Int_t bin);
        virtual Float_t GetErrorY(Int_t bin);
        TF1            *GetFunction(const char *name);
        TH1F           *GetHistogram();
        TList          *GetListOfFunctions() { return fFunctions; }
        Int_t           GetN() {return fNpoints;}
        Float_t        *GetX() {return fX;}
        Float_t        *GetY() {return fY;}
        TAxis          *GetXaxis();
        TAxis          *GetYaxis();
        virtual void    GetPoint(Int_t i, Float_t &x, Float_t &y);
        virtual void    InitExpo();
        virtual void    InitGaus();
        virtual void    InitPolynom();
        virtual void    Paint(Option_t *chopt="");
        virtual void    PaintGraph(Int_t npoints, Float_t *x, Float_t *y, Option_t *option="");
        virtual void    PaintGrapHist(Int_t npoints, Float_t *x, Float_t *y, Option_t *option="");
        virtual void    Print(Option_t *chopt="");
        static  void    RemoveFunction(TGraph *gr, TObject *obj);
        virtual void    SavePrimitive(ofstream &out, Option_t *option);
        virtual void    SetMaximum(Float_t maximum=-1111); // *MENU*
        virtual void    SetMinimum(Float_t minimum=-1111); // *MENU*
        virtual void    SetPoint(Int_t i, Float_t x, Float_t y);
        virtual void    SetTitle(const char *title="");    // *MENU*
                void    Smooth(Int_t npoints, Float_t *x, Float_t *y, Int_t drawtype);
                void    Zero(Int_t &k,Float_t AZ,Float_t BZ,Float_t E2,Float_t &X,Float_t &Y
                        ,Int_t maxiterations);

        ClassDef(TGraph,1)  //Graph graphics class
};

#endif


