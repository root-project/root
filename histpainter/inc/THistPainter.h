// @(#)root/histpainter:$Name$:$Id$
// Author: Rene Brun   26/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_THistPainter
#define ROOT_THistPainter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THistPainter                                                         //
//                                                                      //
// helper class to draw histograms                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualHistPainter
#include "TVirtualHistPainter.h"
#endif

#ifndef ROOT_TH1
#include "TH1.h"
#endif

class TGaxis;
class TLego;

class THistPainter : public TVirtualHistPainter {

protected:
    TH1        *fH;               //pointer to histogram to paint
    TAxis      *fXaxis;           //pointer to X axis
    TAxis      *fYaxis;           //pointer to Y axis
    TAxis      *fZaxis;           //pointer to Z axis
    TList      *fFunctions;       //pointer to histogram list of functions
    TLego      *fLego;            //pointer to a TLego object
    Float_t    *fXbuf;            //X buffer coodinates
    Float_t    *fYbuf;            //Y buffer coodinates
    Int_t       fNIDS;            //Number of stacked histograms

public:
    THistPainter();
    virtual ~THistPainter();
    virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
    virtual void    DrawPanel();
    virtual void    ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual void    FitPanel();
    virtual char   *GetObjectInfo(Int_t px, Int_t py);
    virtual Int_t   MakeChopt(Option_t *option);
    virtual void    Paint(Option_t *option="");
    virtual void    PaintArrows();
    virtual void    PaintAxis();
    virtual void    PaintBoxes();
    virtual void    PaintColorLevels();
    virtual void    PaintContour();
    virtual Int_t   PaintContourLine(Float_t elev1, Int_t icont1, Float_t x1, Float_t y1,
                           Float_t elev2, Int_t icont2, Float_t x2, Float_t y2,
                           Float_t *xarr, Float_t *yarr, Int_t *itarr);
    virtual void    PaintErrors();
    virtual void    PaintFrame();
    virtual void    PaintFunction();
    virtual void    PaintHist();
    virtual void    PaintH3(Option_t *option="");
    virtual Int_t   PaintInit();
    virtual void    PaintLego();
    virtual void    PaintLegoAxis(TGaxis *axis, Float_t ang);
    virtual void    PaintPalette();
    virtual void    PaintScatterPlot();
    virtual void    PaintStat(Int_t dostat, TF1 *fit);
    virtual void    PaintStat2(Int_t dostat, TF1 *fit);
    virtual void    PaintSurface();
    virtual void    PaintTable();
    virtual void    PaintText();
    virtual void    PaintTitle();
    virtual void    RecalculateRange();
    virtual void    SetHistogram(TH1 *h);
    virtual Int_t   TableInit();

    ClassDef(THistPainter,0)  //helper class to draw histograms
};

#endif
