// @(#)root/treeviewer:$Name:  $:$Id: TParallelCoordVar.h,v 1.1 2007/07/24 20:00:46 brun Exp $
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParallelCoordVar
#define ROOT_TParallelCoordVar

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParallelCoordVar                                                    //
//                                                                      //
// Class containing a variable for the TParallelCoord.                  //
//                                                                      //
// Options can be defined on an axis using the right mouse              //
// click. These options can be applied to every axes using the editor.  //
//    - Axis width: If set to 0, the axis is simply a line. If higher,  //
//      a color histogram is drawn on the axis.                         //
//    - Axis histogram height: If not 0, a usual bar histogram is drawn //
//      on the plot.                                                    //
// The order in which the variables are drawn is essential to see the   //
// clusters. The axes can be dragged to change their position.          //
// A zoom is also available. The logarithm scale is also available by   //
// right clicking on the axis.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

class TParallelCoord;
class TParallelCoordSelect;
class TH1F;

class TParallelCoordVar : public TNamed, public TAttLine, public TAttFill {
public:
   enum {
      kLogScale      =BIT(14),
      kShowBox       =BIT(15),
      kShowBarHisto  =BIT(16)
   };
private:
   Int_t             fNbins;        // Number of bins in fHistogram.
   Int_t             fHistoLW;      // Line width used to draw the histogram line.
   Int_t             fId;           // Id identifying the variable for the editor.
   Long64_t          fNentries;      // Number of stored entries values.
   Double_t          fX1;           // x1 coordinate of the axis.
   Double_t          fX2;           // x2 coordinate of the axis.
   Double_t          fY1;           // y1 coordinate of the axis.
   Double_t          fY2;           // y2 coordinate of the axis.
   Double_t          fMinInit;      // Memory of the minimum when first initialized.
   Double_t          fMaxInit;      // Memory of the maximum when first initialized.
   Double_t          fMean;         // Average.
   Double_t          fMinCurrent;   // Current used minimum.
   Double_t          fMaxCurrent;   // Current used maximum.
   Double_t          fMed;          // Median value (Q2).
   Double_t          fQua1;         // First quantile (Q1).
   Double_t          fQua3;         // Third quantile (Q3).
   Double_t          fHistoHeight;  // Histogram Height.
   Double_t         *fVal;          //[fNentries] Entries values for the variable.
   TList            *fRanges;       // List of the TParallelRange owned by TParallelCoordVar.
   TParallelCoord   *fParallel;     //-> Pointer to the TParallelCoord which owns the TParallelCoordVar.
   TH1F             *fHistogram;    // Histogram holding the variable distribution.

public:
   TParallelCoordVar();
   TParallelCoordVar(Double_t *val, const char* title,Int_t id, TParallelCoord* gram);
   ~TParallelCoordVar();

   void           AddRange(); // *MENU*
   void           DeleteVariable(); // *MENU*
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   Draw(Option_t *option="");
   Bool_t         Eval(Long64_t evtidx, TParallelCoordSelect *select); // Check an entry is within its ranges owned by a given TParallelSelect.
   virtual void   ExecuteEvent(Int_t entry, Int_t px, Int_t py);
   Bool_t         GetBarHisto() {return TestBit(kShowBarHisto);}
   Bool_t         GetBoxPlot() {return TestBit(kShowBox);}
   TH1F          *GetHistogram();
   Int_t          GetId() {return fId;}
   Bool_t         GetLogScale() const {return TestBit (kLogScale);}
   Int_t          GetHistBinning() const {return fNbins;}
   Double_t       GetCurrentMin() const {return fMinCurrent;}
   Double_t       GetCurrentMax() const {return fMaxCurrent;}
   Double_t       GetCurrentAverage() const {return fMean;}
   void           GetEntryXY(Long64_t n, Double_t & x, Double_t & y);
   Int_t          GetEvtWeight(Long64_t evtidx);
   Double_t       GetHistHeight() {return fHistoHeight;}
   Int_t          GetHistLineWidth() {return fHistoLW;}
   void           GetQuantiles();
   Double_t       GetX() {return fX1;}
   Double_t       GetY() {return fY1;}
   Int_t          GetNbins() {return fNbins;}
   Long64_t       GetNentries() const {return fNentries;}
   virtual char  *GetObjectInfo(Int_t px, Int_t py) const;
   TParallelCoord* GetParallel() {return fParallel;}
   TList         *GetRanges() {return fRanges;}
   Double_t      *GetValues() {return fVal;}
   Double_t       GetValuefromXY(Double_t x,Double_t y);
   Bool_t         GetVert() {return fX1 == fX2;} // Tells if the axis is vertical or not.
   void           GetXYfromValue(Double_t value, Double_t & x, Double_t & y);
   void           Init();
   virtual void   Paint(Option_t* option="");
   void           PaintBoxPlot();
   void           PaintHistogram();
   void           PaintLabels();
   virtual void   Print(Option_t* option="") const; // *MENU*
   void           SetBoxPlot(Bool_t box); // *TOGGLE* *GETTER=GetBoxPlot
   void           SetBarHisto(Bool_t h) {SetBit(kShowBarHisto,h);} // *TOGGLE* *GETTER=GetBarHisto
   void           SetHistogramLineWidth(Int_t lw=2) {fHistoLW = lw;} // *MENU*
   void           SetHistogramHeight(Double_t h=0); // *MENU*
   void           SetHistogramBinning(Int_t n=100); // *MENU*
   void           SetCurrentLimits(Double_t min, Double_t max); // *MENU*
   void           SetCurrentMin(Double_t min);
   void           SetCurrentMax(Double_t max);
   void           SetLiveRangesUpdate(Bool_t on);
   void           SetLogScale(Bool_t log); // *TOGGLE* *GETTER=GetLogScale
   //void           SetQuantiles(Int_t length, Long64_t *indexes);
   void           SetX(Double_t x, Bool_t gl);    // Set a new x position in case of a vertical display.
   void           SetY(Double_t y, Bool_t gl);    // Set a new y position in case of a horizontal display.
   void           Unzoom() {SetCurrentLimits(fMinInit,fMaxInit);} // *MENU* Reset fMin and fMax to their original value.

   ClassDef(TParallelCoordVar,0); // A Variable of a parallel coordinates plot.
};

#endif
