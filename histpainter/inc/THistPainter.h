// @(#)root/histpainter:$Name:  $:$Id: THistPainter.h,v 1.11 2002/03/21 16:15:43 rdm Exp $
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
#ifndef ROOT_TCutG
#include "TCutG.h"
#endif

class TGaxis;
class TPainter3dAlgorithms;
class TPaveStats;
const Int_t kMaxCuts = 16;

class THistPainter : public TVirtualHistPainter {

protected:
    TH1                  *fH;                 //pointer to histogram to paint
    TAxis                *fXaxis;             //pointer to X axis
    TAxis                *fYaxis;             //pointer to Y axis
    TAxis                *fZaxis;             //pointer to Z axis
    TList                *fFunctions;         //pointer to histogram list of functions
    TPainter3dAlgorithms *fLego;              //pointer to a TPainter3dAlgorithms object
    Double_t             *fXbuf;              //X buffer coordinates
    Double_t             *fYbuf;              //Y buffer coordinates
    Int_t                 fNcuts;             //Number of graphical cuts
    Int_t                 fCutsOpt[kMaxCuts]; //sign of each cut
    TCutG                *fCuts[kMaxCuts];    //Pointers to graphical cuts
    TObjArray            *fStack;             //Pointer to stack of histograms (if any)
    TPaveStats           *fStats;             //Pointer to the current stats pave (if any)

public:
    THistPainter();
    virtual ~THistPainter();
    virtual void       DefineColorLevels(Int_t ndivz);
    virtual Int_t      DistancetoPrimitive(Int_t px, Int_t py);
    virtual void       DrawPanel();
    virtual void       ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual void       FitPanel();
    virtual char      *GetObjectInfo(Int_t px, Int_t py) const;
    virtual TObjArray *GetStack() const {return fStack;}
    virtual TPaveStats*GetStats() const {return fStats;}
    virtual Bool_t     IsInside(Int_t x, Int_t y);
    virtual Bool_t     IsInside(Double_t x, Double_t y);
    virtual Int_t      MakeChopt(Option_t *option);
    virtual Int_t      MakeCuts(char *cutsopt);
    virtual void       Paint(Option_t *option="");
    virtual void       PaintArrows(Option_t *option);
    virtual void       PaintAxis();
    virtual void       PaintBar(Option_t *option);
    virtual void       PaintBarH(Option_t *option);
    virtual void       PaintBoxes(Option_t *option);
    virtual void       PaintColorLevels(Option_t *option);
    virtual void       PaintContour(Option_t *option);
    virtual Int_t      PaintContourLine(Double_t elev1, Int_t icont1, Double_t x1, Double_t y1,
                           Double_t elev2, Int_t icont2, Double_t x2, Double_t y2,
                           Double_t *xarr, Double_t *yarr, Int_t *itarr, Double_t *levels);
    virtual void       PaintErrors(Option_t *option);
    virtual void       PaintFrame();
    virtual void       PaintFunction(Option_t *option);
    virtual void       PaintHist(Option_t *option);
    virtual void       PaintH3(Option_t *option="");
    virtual void       PaintH3Iso();
    virtual Int_t      PaintInit();
    virtual Int_t      PaintInitH();
    virtual void       PaintLego(Option_t *option);
    virtual void       PaintLegoAxis(TGaxis *axis, Double_t ang);
    virtual void       PaintPalette();
    virtual void       PaintScatterPlot(Option_t *option);
    virtual void       PaintStat(Int_t dostat, TF1 *fit);
    virtual void       PaintStat2(Int_t dostat, TF1 *fit);
    virtual void       PaintSurface(Option_t *option);
    virtual void       PaintTable(Option_t *option);
    virtual void       PaintText(Option_t *option);
    virtual void       PaintTitle();
    virtual void       RecalculateRange();
    virtual void       RecursiveRemove(TObject *);
    virtual void       SetHistogram(TH1 *h);
    virtual void       SetStack(TObjArray *stack) {fStack = stack;}
    virtual Int_t      TableInit();

    ClassDef(THistPainter,0)  //Helper class to draw histograms
};

#endif
