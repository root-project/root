// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  20/07/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSpider
#define ROOT_TSpider

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpider                                                              //
//                                                                      //
// TSpider is a manager used to paint a spider view                     //
// of the events of a TNtuple.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TAttFill.h"
#include "TAttLine.h"

class TTree;
class TGraphPolargram;
class TPolyLine;
class TTreeFormula;
class TTreeFormulaManager;
class TList;
class TSelectorDraw;
class TString;
class TLatex;
class TCanvas;
class TArc;

class TSpider : public TObject, public TAttFill, public TAttLine {
private:
   UInt_t                  fNx; // Number of horizontal spider plots.
   UInt_t                  fNy; // Number of vertical spider plots.
   UInt_t                  fNcols; // Number of variables.
   Int_t                   fArraySize; // Actual size of the arrays.
   Long64_t                fEntry; // Present entry number in fTree.
   Long64_t                fNentries; // Number of entries.
   Long64_t                fFirstEntry; // First entry.
   Long64_t*               fCurrentEntries; //![fNx*fNy] current selected entries;
   Double_t*               fAve; //[fNcols] Average value of each variable.
   Double_t*               fMax; //[fNcols]  Maximum value of the variables.
   Double_t*               fMin; //[fNcols]  Minimum value of the variables.
   TList*                  fSuperposed; // Superposed spider plots.
   TTree*                  fTree; // Pointer to the TTree to represent.
   TPolyLine*              fAveragePoly; // Polygon representing the average variables value.
   TArc**                  fAverageSlices; //! Average slices.
   TCanvas*                fCanvas; //! Pointer to the mother pad.
   TList*                  fFormulas; // List of all formulas to represent.
   TList*                  fInput; // Used for fSelector.
   TTreeFormulaManager*    fManager; // Coordinator for the formulas.
   TGraphPolargram*        fPolargram; // Polar graph.
   TList*                  fPolyList; // Polygons representing the variables.
   TTreeFormula*           fSelect; // Selection condition
   TSelectorDraw*          fSelector; //! Selector.
   Bool_t                  fAngularLabels; // True if the labels are oriented according to their axis.
   Bool_t                  fDisplayAverage; // Display or not the average.
   Bool_t                  fForceDim; // Force dimension.
   Bool_t                  fSegmentDisplay; // True if displaying a segment plot.
   Bool_t                  fShowRange; // Show range of variables or not.

   Int_t          FindTextAlign(Double_t theta);
   Double_t       FindTextAngle(Double_t theta);
   void           InitVariables(Long64_t firstentry, Long64_t nentries);
   void           DrawPoly(Option_t* options);
   void           DrawPolyAverage(Option_t* options);
   void           DrawSlices(Option_t* options);
   void           DrawSlicesAverage(Option_t* options);
   void           SyncFormulas();
   void           InitArrays(Int_t newsize);
   void           SetCurrentEntries();
   void           UpdateView();

public:
   TSpider();
   TSpider(TTree* tree, const char *varexp, const char *selection, Option_t *option="",
                  Long64_t nentries=0, Long64_t firstentry=0);
   ~TSpider();
   void           AddSuperposed(TSpider* sp);
   void           AddVariable(const char* varexp); // *MENU*
   void           DeleteVariable(const char* varexp); // *MENU*
   virtual void   Draw(Option_t *options="");
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
   Style_t        GetAverageLineStyle() const;
   Color_t        GetAverageLineColor() const;
   Width_t        GetAverageLineWidth() const;
   Color_t        GetAverageFillColor() const;
   Style_t        GetAverageFillStyle() const;
   Bool_t         GetDisplayAverage() const {return fDisplayAverage;}
   Long64_t       GetCurrentEntry() const {return fEntry;}
   Long64_t       GetEntriesToProcess(Long64_t firstentry, Long64_t nentries) const;
   Int_t          GetNx() const {return fNx;}
   Int_t          GetNy() const {return fNy;}
   Bool_t         GetSegmentDisplay() const {return fSegmentDisplay;}
   void           GotoEntry(Long64_t e); // *MENU*
   void           GotoNext(); // *MENU*
   void           GotoPrevious(); // *MENU*
   void           GotoFollowing(); // *MENU*
   void           GotoPreceding(); // *MENU*
   virtual void   Paint(Option_t *options);
   void           SetAverageLineStyle(Style_t sty);
   void           SetAverageLineColor(Color_t col);
   void           SetAverageLineWidth(Width_t wid);
   void           SetAverageFillColor(Color_t col);
   void           SetAverageFillStyle(Style_t sty);
   virtual void   SetLineStyle(Style_t sty);
   virtual void   SetLineColor(Color_t col);
   virtual void   SetLineWidth(Width_t wid);
   virtual void   SetFillColor(Color_t col);
   virtual void   SetFillStyle(Style_t sty);
   void           SetDisplayAverage(Bool_t disp); // *TOGGLE*
   void           SetVariablesExpression(const char* varexp);
   void           SetNdivRadial(Int_t div); // *MENU*
   void           SetNx(UInt_t nx); // *MENU*
   void           SetNy(UInt_t ny); // *MENU*
   void           SetSelectionExpression(const char* selexp);
   void           SetSegmentDisplay(Bool_t seg); // *TOGGLE*
   void           SetShowRange(Bool_t showrange) {fShowRange = showrange;}
   void           SuperposeTo(TSpider* sp) {sp->AddSuperposed(this);}

   ClassDef(TSpider,0)  //Helper class to draw spider
};

#endif
