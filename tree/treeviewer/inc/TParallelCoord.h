// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParallelCoord
#define ROOT_TParallelCoord

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TTree;
class TPaveText;
class TEntryList;
class TParallelCoordSelect;
class TParallelCoordVar;
class TParallelCoordRange;
class TList;
class TGaxis;
class TSelectorDraw;

class TParallelCoord : public TNamed {
public:
   enum {
      kVertDisplay      =BIT(14),      // If the axes are drawn vertically, false if horizontally.
      kCurveDisplay     =BIT(15),      // If the polylines are replaced by interpolated curves.
      kPaintEntries     =BIT(16),      // To prentry the TParallelCoord to paint all the entries.
      kLiveUpdate       =BIT(17),      // To paint the entries when being modified.
      kGlobalScale      =BIT(19),      // Every variable is on the same scale.
      kCandleChart      =BIT(20),      // To produce a candle chart.
      kGlobalLogScale   =BIT(21)       // Every variable in log scale.
   };

private:
   UInt_t          fNvar;              // Number of variables.
   Long64_t        fCurrentFirst;      // First entry to display.
   Long64_t        fCurrentN;          // Number of entries to display.
   Long64_t        fNentries;          // Number of entries;
   Int_t           fDotsSpacing;       // Spacing between dots to draw the entries.
   Color_t         fLineColor;         // entries line color.
   Width_t         fLineWidth;         // entries line width.
   Int_t           fWeightCut;         // Specify a cut on the entries from their weight (see TParallelCoordVar::GetEvtWeight(Long64_t))
   TEntryList     *fCurrentEntries;    //-> Current selected entries in the tree.
   TEntryList     *fInitEntries;       //-> Selected entries when TParallelCoord first initialized.
   TTree          *fTree;              //! Pointer to the TTree.
   TString         fTreeName;          // Name of the tree.
   TString         fTreeFileName;      // Name of the file containing the tree.
   TList          *fVarList;           // List of the variables.
   TList          *fSelectList;        // List of selections over the variables.
   TParallelCoordSelect* fCurrentSelection; //! Current Selection being edited.
   TGaxis         *fCandleAxis;        //! An axis used when displaying a candle chart.

   void            Init();
   void            PaintEntries(TParallelCoordSelect* sel=NULL);
   void            SetAxesPosition();

public:
   TParallelCoord();
   TParallelCoord(Long64_t nentries);
   TParallelCoord(TTree* tree, Long64_t nentries);
   ~TParallelCoord();

   void           AddVariable(Double_t* val, const char* title="");
   void           AddVariable(const char* varexp);
   void           AddSelection(const char* title);
   void           ApplySelectionToTree(); // *MENU*
   static void    BuildParallelCoord(TSelectorDraw* selector, Bool_t candle);
   void           CleanUpSelections(TParallelCoordRange* range);
   void           RemoveVariable(TParallelCoordVar* var);
   TParallelCoordVar* RemoveVariable(const char* var);
   void           DeleteSelection(TParallelCoordSelect* sel);
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   Draw(Option_t* options="");
   virtual void   ExecuteEvent(Int_t entry, Int_t px, Int_t py);
   Bool_t         GetCandleChart() {return TestBit(kCandleChart);}
   Long64_t       GetCurrentFirst() {return fCurrentFirst;}
   Long64_t       GetCurrentN() {return fCurrentN;}
   TParallelCoordSelect* GetCurrentSelection();
   Bool_t         GetCurveDisplay() const {return TestBit(kCurveDisplay);}
   Int_t          GetDotsSpacing() const {return fDotsSpacing;}
   TEntryList    *GetEntryList(Bool_t sel=kTRUE);
   Double_t       GetGlobalMin();
   Double_t       GetGlobalMax();
   Bool_t         GetGlobalScale() {return TestBit(kGlobalScale);}
   Bool_t         GetGlobalLogScale() {return TestBit(kGlobalLogScale);}
   Color_t        GetLineColor() {return fLineColor;}
   Width_t        GetLineWidth() {return fLineWidth;}
   Int_t          GetNbins();
   UInt_t         GetNvar() {return fNvar;}
   Long64_t       GetNentries() {return fNentries;}
   TList         *GetSelectList() {return fSelectList;}
   TParallelCoordSelect* GetSelection(const char* title);
   TTree         *GetTree();
   Double_t      *GetVariable(const char* var);
   Double_t      *GetVariable(Int_t i);
   TList         *GetVarList() {return fVarList;}
   Bool_t         GetVertDisplay() const {return TestBit(kVertDisplay);}
   Int_t          GetWeightCut() const {return fWeightCut;};
   virtual void   Paint(Option_t* options="");
   void           ResetTree();
   void           SaveEntryLists(const char* filename="", Bool_t overwrite=kFALSE); // *MENU*
   void           SavePrimitive(ostream & out,Option_t *options);
   void           SaveTree(const char* filename="", Bool_t overwrite=kFALSE); // *MENU*
   void           SetAxisHistogramBinning(Int_t n=100); // *MENU*
   void           SetAxisHistogramHeight(Double_t h=0.5); // *MENU*
   void           SetAxisHistogramLineWidth(Int_t lw=2); // *MENU*
   void           SetCandleChart(Bool_t can); // *TOGGLE* *GETTER=GetCandleChart
   virtual void   SetCurveDisplay(Bool_t curve=1) {SetBit(kCurveDisplay,curve);} // *TOGGLE* *GETTER=GetCurveDisplay
   void           SetCurrentEntries(TEntryList* entries) {fCurrentEntries = entries;}
   void           SetCurrentFirst(Long64_t);
   void           SetCurrentN(Long64_t);
   TParallelCoordSelect* SetCurrentSelection(const char* title);
   void           SetCurrentSelection(TParallelCoordSelect* sel);
   void           SetDotsSpacing(Int_t s=0); // *MENU*
   static void    SetEntryList(TParallelCoord* para, TEntryList* enlist);
   void           SetGlobalScale(Bool_t gl); // *TOGGLE* *GETTER=GetGlobalScale
   void           SetGlobalLogScale(Bool_t); // *TOGGLE* *GETTER=GetGlobalLogScale
   void           SetGlobalMin(Double_t min);
   void           SetGlobalMax(Double_t max);
   void           SetInitEntries(TEntryList* entries) {fInitEntries = entries;}
   void           SetLineColor(Color_t col) {fLineColor = col;}
   void           SetLineWidth(Width_t wid) {fLineWidth = wid;}
   void           SetLiveRangesUpdate(Bool_t);
   void           SetNentries(Long64_t n) {fNentries = n;}
   void           SetTree(TTree* tree) {fTree = tree;}
   void           SetVertDisplay(Bool_t vert=kTRUE); // *TOGGLE* *GETTER=GetVertDisplay
   void           SetWeightCut(Int_t w=0) {fWeightCut = w;} // *MENU*
   void           UnzoomAll(); // *MENU*

   ClassDef(TParallelCoord,1); // To display parallel coordinates plots.
};

#endif
