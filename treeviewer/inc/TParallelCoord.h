// @(#)root/treeviewer:$Name:  $:$Id: TParallelCoord.h,v 1.1 2007/07/24 20:00:46 brun Exp $
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParallel
#define ROOT_TParallel

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParallelCoord                                                       //
//                                                                      //
// The multidimensional system of Parallel coordinates is a common      //
// way of studying high-dimensional geometry and visualizing            //
// multivariate problems.                                               //
//                                                                      //
// To show a set of points in an n-dimensional space, a backdrop is     //
// drawn consisting of n parallel lines. A point in n-dimensional space //
// is represented as a polyline with vertices on the parallel axes;     //
// the position of the vertex on the i-th axis corresponds to the i-th  //
// coordinate of the point.                                             //
//                                                                      //
// This tool comes with a rather large gui in the editor. It is         //
// necessary to use this editor in order to explore a data set, as      //
// explained below.                                                     //
//                                                                      //
// Reduce cluttering:                                                   //
//                                                                      //
// The main issue for parallel coordinates is the very high cluttering  //
// of the output when dealing with large data set. Two techniques have  //
// been implemented to bypass that so far:                              //
//    - Draw doted lines instead of plain lines with an adjustable      //
//      dots spacing. A slider to adjust the dots spacing is available  //
//      in the editor.                                                  //
//    - Sort the entries to display with  a "weight cut". On each axis  //
//      is drawn a histogram describing the distribution of the data    //
//      on the corresponding variable. The "weight" of an entry is the  //
//      sum of the bin content of each bin the entry is going through.  //
//      An entry going through the histograms peaks will have a big     //
//      weight wether an entry going randomly through the histograms    //
//      will have a rather small weight. Setting a cut on this weight   //
//      allows to draw only the most representative entries. A slider   //
//      set the cut is also available in the gui.                       //
//                                                                      //
// Selections:                                                          //
//                                                                      //
// Selections of specific entries can be defined over the data set      //
// using parallel coordinates. With that representation, a selection is //
// an ensemble of ranges defined on the axes. Ranges defined on the     //
// same axis are conjugated with OR (an entry must be in one or the     //
// other ranges to be selected). Ranges on different axes are           //
// are conjugated with AND (an entry must be in all the ranges to be    //
// selected).                                                           //
// Several selections can be defined with different colors. It is       //
// possible to generate an entry list from a given selection and apply  //
// it to the tree using the editor ("Apply to tree" button).            //
//                                                                      //
// Axes:                                                                //
//                                                                      //
// Options can be defined each axis separatly using the right mouse     //
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
// Candle chart:                                                        //
//                                                                      //
// TParallelCoord can also be used to display a candle chart. In that   //
// mode, every variable is drawn in the same scale. The candle chart    //
// can be combined with the parallel coordinates mode, drawing the      //
// candle sticks over the axes.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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

class TParallelCoord : public TNamed, public TAttLine {
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
   Long64_t        fNentries;           // Number of entries;
   Int_t           fDotsSpacing;       // Spacing between dots to draw the entries.
   Int_t           fWeightCut;         // Specify a cut on the entries from their weight (see TParallelCoordVar::GetEvtWeight(Long64_t))
   TEntryList     *fEntries;           // Selected entries in the tree.
   TTree          *fTree;              // Pointer to the TTree.
   TList          *fVarList;           // List of the variables.
   TList          *fSelectList;        // List of selections over the variables.
   TParallelCoordSelect* fCurrentSelection; // Current Selection being edited.
   TGaxis         *fCandleAxis;        // An axis used when displaying a candle chart.

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
   static void    BuildParallelCoord(TObject** pobj, TTree* tree, Int_t dim, Long64_t nentries,
                                     Double_t **val, const char* title, TString* var, Bool_t candle);
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
   TParallelCoordSelect* GetCurrentSelection() {return fCurrentSelection;}
   Bool_t         GetCurveDisplay() const {return TestBit(kCurveDisplay);}
   Int_t          GetDotsSpacing() const {return fDotsSpacing;}
   TEntryList    *GetEntryList(Bool_t sel=kTRUE);
   Double_t       GetGlobalMin();
   Double_t       GetGlobalMax();
   Bool_t         GetGlobalScale() {return TestBit(kGlobalScale);}
   Bool_t         GetGlobalLogScale() {return TestBit(kGlobalLogScale);}
   Int_t          GetNbins();
   UInt_t         GetNvar() {return fNvar;}
   Long64_t       GetNentries() {return fNentries;}
   TList         *GetSelectList() {return fSelectList;}
   TTree         *GetTree() {return fTree;}
   Double_t      *GetVariable(const char* var);
   Double_t      *GetVariable(Int_t i);
   TList         *GetVarList() {return fVarList;}
   Bool_t         GetVertDisplay() const {return TestBit(kVertDisplay);}
   Int_t          GetWeightCut() const {return fWeightCut;};
   virtual void   Paint(Option_t* options="");
   void           SetAxisHistogramBinning(Int_t n=100); // *MENU*
   void           SetAxisHistogramHeight(Double_t h=0.5); // *MENU*
   void           SetAxisHistogramLineWidth(Int_t lw=2); // *MENU*
   void           SetCandleChart(Bool_t can); // *TOGGLE* *GETTER=GetCandleChart
   virtual void   SetCurveDisplay(Bool_t curve=1) {SetBit(kCurveDisplay,curve);} // *TOGGLE* *GETTER=GetCurveDisplay
   void           SetCurrentFirst(Long64_t);
   void           SetCurrentN(Long64_t);
   void           SetCurrentSelection(const char* title);
   void           SetCurrentSelection(TParallelCoordSelect* sel);
   void           SetDotsSpacing(Int_t s=0); // *MENU*
   void           SetEntryList(TEntryList* enlist) {fEntries = enlist;}
   static void    SetEntryList(TParallelCoord* para, TEntryList* enlist) {para->SetEntryList(enlist);}
   void           SetGlobalScale(Bool_t gl); // *TOGGLE* *GETTER=GetGlobalScale
   void           SetGlobalLogScale(Bool_t); // *TOGGLE* *GETTER=GetGlobalLogScale
   void           SetGlobalMin(Double_t min);
   void           SetGlobalMax(Double_t max);
   void           SetLiveRangesUpdate(Bool_t);
   void           SetNentries(Long64_t n) {fNentries = n;}
   void           SetTree(TTree* tree) {fTree = tree;}
   void           SetVertDisplay(Bool_t vert=kTRUE); // *TOGGLE* *GETTER=GetVertDisplay
   void           SetWeightCut(Int_t w=0) {fWeightCut = w;} // *MENU*
   void           UnzoomAll(); // *MENU*

   ClassDef(TParallelCoord,0); // To display parallel coordinates plots.
};

#endif
