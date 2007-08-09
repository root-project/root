// @(#)root/treeviewer:$Name:  $:$Id: TParallelCoord.cxx,v 1.1 2007/08/08 12:57:38 brun Exp $
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TParallelCoord.h"
#include "TParallelCoordVar.h"
#include "TParallelCoordRange.h"

#include "Riostream.h"
#include "TROOT.h"
#include "TVirtualX.h"
#include "TPad.h"
#include "TPolyLine.h"
#include "TGraph.h"
#include "TPaveText.h"
#include "float.h"
#include "TMath.h"
#include "TBox.h"
#include "TH1.h"
#include "TStyle.h"
#include "TEntryList.h"
#include "TFrame.h"
#include "TTree.h"
#include "TTreePlayer.h"
#include "TSelectorDraw.h"
#include "TTreeFormula.h"
#include "TView.h"
#include "TRandom.h"
#include "TEnv.h"
#include "TCanvas.h"
#include "TGaxis.h"

ClassImp(TParallelCoord)

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


//______________________________________________________________________________
TParallelCoord::TParallelCoord()
   :TNamed()
{
   // Default constructor.

   Init();
}


//______________________________________________________________________________
TParallelCoord::TParallelCoord(Long64_t nentries)
{
   // Constructor without a reference to a tree,
   // the datas must be added afterwards with
   // TParallelCoord::AddVariable(Double_t*,const char*).

   Init();
   fNentries = nentries;
   fCurrentN = fNentries;
   fVarList = new TList();
   fSelectList = new TList();
   fCurrentSelection = new TParallelCoordSelect();
   fSelectList->Add(fCurrentSelection);
}


//______________________________________________________________________________
TParallelCoord::TParallelCoord(TTree* tree, Long64_t nentries)
   :TNamed("Graph","Graph")
{
   // Normal constructor, the datas must be added afterwards
   // with TParallelCoord::AddVariable(Double_t*,const char*).

   Init();
   fNentries = nentries;
   fCurrentN = fNentries;
   fTree = tree;
   fVarList = new TList();
   fSelectList = new TList();
   fCurrentSelection = new TParallelCoordSelect();
   fSelectList->Add(fCurrentSelection);
}


//______________________________________________________________________________
TParallelCoord::~TParallelCoord()
{
   // Destructor.

   if (fEntries) delete fEntries;
   if (fVarList) {
      fVarList->Delete();
      delete fVarList;
   }
   if (fSelectList) {
      fSelectList->Delete();
      delete fSelectList;
   }
   if (fCandleAxis) delete fCandleAxis;
   SetDotsSpacing(0);
}


//______________________________________________________________________________
void TParallelCoord::AddVariable(Double_t* val, const char* title)
{
   // Add a variable.

   ++fNvar;
   fVarList->Add(new TParallelCoordVar(val,title,fVarList->GetSize(),this));
   SetAxesPosition();
}


//______________________________________________________________________________
void TParallelCoord::AddVariable(const char* varexp)
{
   // Add a variable from an expression.

   if(!fTree) return; // The tree from whichone one will get the data must be defined.
   
   // Select in the only the entries of this TParallelCoord.
   TEntryList *list = GetEntryList(kFALSE);
   fTree->SetEntryList(list);
   
   // ensure that there is only one variable given:
   
   TString exp = varexp;
   
   if (exp.Contains(':') || exp.Contains(">>") || exp.Contains("<<")) {
      Warning("AddVariable","Only a single variable can be added at a time.");
      return;
   }
   if (exp == ""){
      Warning("AddVariable","Nothing to add");
      return;
   }
   
   Long64_t en = fTree->Draw(varexp,"","goff");
   if (en<0) {
      Warning("AddVariable","%s could not be evaluated",varexp);
      return;
   }
   
   AddVariable(fTree->GetV1(),varexp);
   TParallelCoordVar* var = (TParallelCoordVar*)fVarList->Last();
   var->Draw();
}


//______________________________________________________________________________
void TParallelCoord::AddSelection(const char* title)
{
   // Add a selection.

   TParallelCoordSelect *sel = new TParallelCoordSelect(title);
   fSelectList->Add(sel);
   SetCurrentSelection(sel);
}


//______________________________________________________________________________
void  TParallelCoord::ApplySelectionToTree()
{
   // Apply the current selection to the tree.

   if(!fTree) return;
   TEntryList* enlist = GetEntryList();
   fTree->SetEntryList(enlist);
}


//______________________________________________________________________________
void  TParallelCoord::BuildParallelCoord(TSelectorDraw* selector, Bool_t candle)
{
   // Call constructor and add the variables.

   TParallelCoord* pc = new TParallelCoord(selector->GetTree(),selector->GetNfill());
   selector->SetObject(pc);
   TString varexp = "";
   for(Int_t i=0;i<selector->GetDimension();++i) {
      pc->AddVariable(selector->GetVal(i),selector->GetVar(i)->GetTitle());
      varexp.Append(Form(":%s",selector->GetVar(i)->GetTitle()));
   }
   varexp.Remove(TString::kLeading,':');
   if (selector->GetSelect()) varexp.Append(Form("{%s}",selector->GetSelect()->GetTitle()));
   pc->SetTitle(varexp.Data());
   if (!candle) pc->Draw();
   else pc->Draw("candle");
}

//______________________________________________________________________________
void TParallelCoord::CleanUpSelections(TParallelCoordRange* range)
{
   // Clean up the selections from the ranges which could have been deleted
   // when a variable has been deleted.

   TIter next(fSelectList);
   while(TParallelCoordSelect* select = (TParallelCoordSelect*)next()){
      if(select->Contains(range)) select->Remove(range);
   }
}


//______________________________________________________________________________
void TParallelCoord::DeleteSelection(TParallelCoordSelect * sel)
{
   // Delete a selection.

   fSelectList->Remove(sel);
   delete sel;
   if(fSelectList->GetSize() == 0) fCurrentSelection = NULL;
   else fCurrentSelection = (TParallelCoordSelect*)fSelectList->At(0);
}


//______________________________________________________________________________
Int_t TParallelCoord::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute the distance from the TParallelCoord.

   if(!gPad) return 9999;
   
   TFrame *frame = gPad->GetFrame();
   
   Double_t x1,x2,y1,y2,xx,yy;
   
   x1 = frame->GetX1()+0.01;
   x2 = frame->GetX2()-0.01;
   y2 = frame->GetY2()-0.01;
   y1 = frame->GetY1()+0.01;
   
   xx = gPad->AbsPixeltoX(px);
   yy = gPad->AbsPixeltoY(py);
   
   if(xx>x1 && xx<x2 && yy>y1 && yy<y2) return 0;
   else                                 return 9999;
}


//______________________________________________________________________________
void TParallelCoord::Draw(Option_t* option)
{
   // Draw the parallel coordinates graph.

   Bool_t optcandle = kFALSE;
   TString opt = option;
   opt.ToLower();
   if(opt.Contains("candle")) {
      optcandle = kTRUE;
      opt.ReplaceAll("candle","");
   }
   if(optcandle) {
      SetBit(kPaintEntries,kFALSE);
      SetBit(kCandleChart,kTRUE);
      SetGlobalScale(kTRUE);
   }
   
   if (gPad) {
      if (!gPad->IsEditable()) gROOT->MakeDefCanvas();
   } else gROOT->MakeDefCanvas();
   TView *view = gPad->GetView();
   if(view){
      delete view;
      gPad->SetView(NULL);
   }
   gPad->Clear();
   if (!optcandle) {
      ((TCanvas*)gPad)->ToggleEditor();
      ((TCanvas*)gPad)->ToggleEventStatus();
   }
   
   gPad->SetBit(TGraph::kClipFrame,kTRUE);
   
   TFrame *frame = new TFrame(0.1,0.1,0.9,0.9);
   frame->SetBorderSize(0);
   frame->SetBorderMode(0);
   frame->SetFillStyle(0);
   frame->SetLineColor(gPad->GetFillColor());
   frame->Draw();
   AppendPad(option);
   TPaveText *title = new TPaveText(0.05,0.95,0.35,1);
   title->AddText(GetTitle());
   title->Draw();
   SetAxesPosition();
   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      if(optcandle) {
         var->SetBoxPlot(kTRUE);
         var->SetHistogramHeight(0.5);
         var->SetHistogramLineWidth(0);
      }
      var->Draw();
   }
   
   if (optcandle) {
      if (TestBit(kVertDisplay)) fCandleAxis = new TGaxis(0.05,0.1,0.05,0.9,GetGlobalMin(),GetGlobalMax());
      else                       fCandleAxis = new TGaxis(0.1,0.05,0.9,0.05,GetGlobalMin(),GetGlobalMax());
      fCandleAxis->Draw();
   }
   
   ((TCanvas*)gPad)->Selected(gPad,this,1);
}


//______________________________________________________________________________
void TParallelCoord::ExecuteEvent(Int_t /*entry*/, Int_t /*px*/, Int_t /*py*/)
{
   // Execute the corresponding entry.

   gPad->SetCursor(kHand);
}


//______________________________________________________________________________
TEntryList* TParallelCoord::GetEntryList(Bool_t sel)
{
   // Get the whole entry list or one for a selection.

   if(!sel){ // If no selection is specified, return the entry list of all the entries.
      return fEntries;
   } else { // return the entry list corresponding to the current selection.
      TEntryList *enlist = new TEntryList(fTree);
      TIter next(fVarList);
      for (Long64_t li=0;li<fEntries->GetN();++li) {
         next.Reset();
         Bool_t inrange=kTRUE;
         while(TParallelCoordVar* var = (TParallelCoordVar*)next()){
            if(!var->Eval(li,fCurrentSelection)) inrange = kFALSE;
         }
         if(!inrange) continue;
         enlist->Enter(fEntries->GetEntry(li));
      }
      return enlist;
   }
}


//______________________________________________________________________________
Double_t TParallelCoord::GetGlobalMax()
{
   // return the global maximum.

   Double_t gmax=-FLT_MAX;
   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      if (gmax < var->GetCurrentMax()) gmax = var->GetCurrentMax();
   }
   return gmax;
}


//______________________________________________________________________________
Double_t TParallelCoord::GetGlobalMin()
{
   // return the global minimum.

   Double_t gmin=FLT_MAX;
   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      if (gmin > var->GetCurrentMin()) gmin = var->GetCurrentMin();
   }
   return gmin;
}


//______________________________________________________________________________
Int_t TParallelCoord::GetNbins()
{
   // get the binning of the histograms.
   
   return ((TParallelCoordVar*)fVarList->First())->GetNbins();
}

      
//______________________________________________________________________________
Double_t* TParallelCoord::GetVariable(const char* vartitle)
{
   // Get the variables values from its title.

   TIter next(fVarList);
   TParallelCoordVar* var = NULL;
   while(((var = (TParallelCoordVar*)next()) != NULL) && (var->GetTitle() != vartitle));
   if(!var) return NULL;
   else     return var->GetValues();
}


//______________________________________________________________________________
Double_t* TParallelCoord::GetVariable(Int_t i)
{
   // Get the variables values from its index.

   if(i<0 || (UInt_t)i>fNvar) return NULL;
   else return ((TParallelCoordVar*)fVarList->At(i))->GetValues();
}


//______________________________________________________________________________
void TParallelCoord::Init()
{
   // Initialise the data members of TParallelCoord.

   fNentries = 0;
   fVarList = NULL;
   fSelectList = NULL;
   SetBit(kVertDisplay,kTRUE);
   SetBit(kCurveDisplay,kFALSE);
   SetBit(kPaintEntries,kTRUE);
   SetBit(kLiveUpdate,kFALSE);
   SetBit(kGlobalScale,kFALSE);
   SetBit(kCandleChart,kFALSE);
   SetBit(kGlobalLogScale,kFALSE);
   fTree = NULL;
   fEntries = NULL;
   fCurrentSelection = NULL;
   fNvar = 0;
   fDotsSpacing = 0;
   fCurrentFirst = 0;
   fCurrentN = 0;
   fCandleAxis = NULL;
   fWeightCut = 0;
   fLineWidth = 1;
   fLineColor = kGreen-8;
}

//______________________________________________________________________________
void TParallelCoord::Paint(Option_t* /*option*/)
{
   // Paint the parallel coordinates graph.

   gPad->PaintPadFrame(0.1,0.1,0.9,0.9);
   TFrame *frame = gPad->GetFrame();
   frame->SetLineColor(gPad->GetFillColor());
   SetAxesPosition();
   if(TestBit(kPaintEntries)){
      PaintEntries(NULL);
      TIter next(fSelectList);
      while(TParallelCoordSelect* sel = (TParallelCoordSelect*)next()) {
         if(sel->GetSize()>0 && sel->TestBit(TParallelCoordSelect::kActivated)) {
            PaintEntries(sel);
         }
      }
   }
   gPad->RangeAxis(0,0,1,1);
}


//______________________________________________________________________________
void TParallelCoord::PaintEntries(TParallelCoordSelect* sel)
{
   // Loop over the entries and paint them.

   if (fVarList->GetSize() < 2) return;
   Int_t i=0;
   Long64_t n=0;
   
   Double_t *x = new Double_t[fNvar];
   Double_t *y = new Double_t[fNvar];
   
   TGraph    *gr     = NULL;
   TPolyLine *pl     = NULL;
   TAttLine  *evline = NULL;
   
   if (TestBit (kCurveDisplay)) {gr = new TGraph(fNvar); evline = (TAttLine*)gr;}
   else                       {pl = new TPolyLine(fNvar); evline = (TAttLine*)pl;}
   
   if (fDotsSpacing == 0) evline->SetLineStyle(1);
   else                   evline->SetLineStyle(11);
   if (!sel){
      evline->SetLineWidth(GetLineWidth());
      evline->SetLineColor(GetLineColor());
   } else {
      evline->SetLineWidth(sel->GetLineWidth());
      evline->SetLineColor(sel->GetLineColor());
   }
   TParallelCoordVar *var;
   
   TFrame *frame = gPad->GetFrame();
   Double_t lx   = ((frame->GetX2() - frame->GetX1())/(fNvar-1));
   Double_t ly   = ((frame->GetY2() - frame->GetY1())/(fNvar-1));
   Double_t a,b;
   TRandom r;
   
   for (n=fCurrentFirst; n<fCurrentFirst+fCurrentN; ++n) {
      TListIter next(fVarList);
      Bool_t inrange = kTRUE;
      // Loop to check whenever the entry must be painted.
      if (sel) {
         while ((var = (TParallelCoordVar*)next())){
            if (!var->Eval(n,sel)) inrange = kFALSE;
         }
      }
      if (fWeightCut > 0) {
         next.Reset();
         Int_t entryweight = 0;
         while ((var = (TParallelCoordVar*)next())) entryweight+=var->GetEntryWeight(n);
         if (entryweight/(Int_t)fNvar < fWeightCut) inrange = kFALSE;
      }
      if(!inrange) continue;
      i = 0;
      next.Reset();
      // Loop to set the polyline points.
      while ((var = (TParallelCoordVar*)next())) {
         var->GetEntryXY(n,x[i],y[i]);
         ++i;
      }
      // beginning to paint the first point at a random distance
      // to avoid artefacts when increasing the dots spacing.
      if (fDotsSpacing != 0) {
         if (TestBit(kVertDisplay)) {
            a    = (y[1]-y[0])/(x[1]-x[0]);
            b    = y[0]-a*x[0];
            x[0] = x[0]+lx*r.Rndm(n);
            y[0] = a*x[0]+b;
         } else {
            a    = (x[1]-x[0])/(y[1]-y[0]);
            b    = x[0]-a*y[0];
            y[0] = y[0]+ly*r.Rndm(n);
            x[0] = a*y[0]+b;
         }
      }
      if (pl) pl->PaintPolyLine(fNvar,x,y);
      else    gr->PaintGraph(fNvar,x,y,"C");
   }
   
   if (pl) delete pl;
   if (gr) delete gr;
   delete [] x;
   delete [] y;
}


//______________________________________________________________________________
void TParallelCoord::RemoveVariable(TParallelCoordVar *var)
{
   // Delete a variable from the graph.

   fVarList->Remove(var);
   fNvar = fVarList->GetSize();
   SetAxesPosition();
}


//______________________________________________________________________________
TParallelCoordVar* TParallelCoord::RemoveVariable(const char* vartitle)
{
   // Delete the variable "vartitle" from the graph.

   TIter next(fVarList);
   TParallelCoordVar* var=NULL;
   while((var = (TParallelCoordVar*)next())) {
      if (!strcmp(var->GetTitle(),vartitle)) break;
   }
   if(!var) Error("RemoveVariable","\"%s\" not a variable",vartitle);
   fVarList->Remove(var);
   fNvar = fVarList->GetSize();
   SetAxesPosition();
   return var;
}


//______________________________________________________________________________
void TParallelCoord::SetAxesPosition()
{
   // Update the position of the axes.

   if(!gPad) return;
   Bool_t vert          = TestBit (kVertDisplay);
   TFrame *frame        = gPad->GetFrame();
   if (fVarList->GetSize() > 1) {
      if (vert) {
         frame->SetX1(1.0/((Double_t)fVarList->GetSize()+1));
         frame->SetX2(1-frame->GetX1());
         frame->SetY1(0.1);
         frame->SetY2(0.9);
         gPad->RangeAxis(1.0/((Double_t)fVarList->GetSize()+1),0.1,1-frame->GetX1(),0.9);
      } else {
         frame->SetX1(0.1);
         frame->SetX2(0.9);
         frame->SetY1(1.0/((Double_t)fVarList->GetSize()+1));
         frame->SetY2(1-frame->GetY1());
         gPad->RangeAxis(0.1,1.0/((Double_t)fVarList->GetSize()+1),0.9,1-frame->GetY1());
      }
      
      Double_t horSpace    = (frame->GetX2() - frame->GetX1())/(fNvar-1);
      Double_t verSpace   = (frame->GetY2() - frame->GetY1())/(fNvar-1);
      Int_t i=0;
      TIter next(fVarList);
      
      while(TParallelCoordVar* var = (TParallelCoordVar*)next()){
         if (vert) var->SetX(gPad->GetFrame()->GetX1() + i*horSpace,TestBit(kGlobalScale));
         else      var->SetY(gPad->GetFrame()->GetY1() + i*verSpace,TestBit(kGlobalScale));
         ++i;
      }
   } else if (fVarList->GetSize()==1) {
      frame->SetX1(0.1);
      frame->SetX2(0.9);
      frame->SetY1(0.1);
      frame->SetY2(0.9);
      if (vert) ((TParallelCoordVar*)fVarList->First())->SetX(0.5,TestBit(kGlobalScale));
      else      ((TParallelCoordVar*)fVarList->First())->SetY(0.5,TestBit(kGlobalScale));
   }
}


//______________________________________________________________________________
void TParallelCoord::SetAxisHistogramBinning(Int_t n)
{
   // Set the same histogram axis binning for all axis.

   TIter next(fVarList);
   while(TParallelCoordVar *var = (TParallelCoordVar*)next()) var->SetHistogramBinning(n);
}


//______________________________________________________________________________
void TParallelCoord::SetAxisHistogramHeight(Double_t h)
{
   // Set the same histogram axis height for all axis.

   TIter next(fVarList);
   while(TParallelCoordVar *var = (TParallelCoordVar*)next()) var->SetHistogramHeight(h);
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalLogScale(Bool_t ls)
{
   // All axes in log scale.

   if (ls == TestBit(kGlobalLogScale)) return;
   SetBit(kGlobalLogScale,ls);
   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) var->SetLogScale(ls);
   if (TestBit(kGlobalScale)) SetGlobalScale(kTRUE);
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalScale(Bool_t gl)
{
   // Constraint all axes to the same scale.

   SetBit(kGlobalScale,gl);
   if (fCandleAxis) {
      delete fCandleAxis;
      fCandleAxis = NULL;
   }
   if (gl) {
      Double_t min,max;
      min = GetGlobalMin();
      max = GetGlobalMax();
      if (TestBit(kGlobalLogScale) && min<=0) min = 0.00001*max;
      if (TestBit(kVertDisplay)) {
         if (!TestBit(kGlobalLogScale)) fCandleAxis = new TGaxis(0.05,0.1,0.05,0.9,min,max);
         else                           fCandleAxis = new TGaxis(0.05,0.1,0.05,0.9,min,max,510,"G");
      } else {
         if (!TestBit(kGlobalLogScale)) fCandleAxis = new TGaxis(0.1,0.05,0.9,0.05,min,max);
         else                           fCandleAxis = new TGaxis(0.1,0.05,0.9,0.05,min,max,510,"G");
      }
      fCandleAxis->Draw();
      SetGlobalMin(min);
      SetGlobalMax(max);
      TIter next(fVarList);
      while (TParallelCoordVar* var = (TParallelCoordVar*)next()) var->GetHistogram();
   }
   gPad->Modified();
   gPad->Update();
}   


//______________________________________________________________________________
void TParallelCoord::SetAxisHistogramLineWidth(Int_t lw)
{
   // Set the same histogram axis line width for all axis.

   TIter next(fVarList);
   while(TParallelCoordVar *var = (TParallelCoordVar*)next()) var->SetHistogramLineWidth(lw);
}


//______________________________________________________________________________
void TParallelCoord::SetCandleChart(Bool_t can)
{
   // Set a candle chart display.

   SetBit(kCandleChart,can);
   SetGlobalScale(can);
   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      var->SetBoxPlot(can);
      var->SetHistogramLineWidth(0);
   }
   if (fCandleAxis) delete fCandleAxis;
   fCandleAxis = NULL;
   SetBit(kPaintEntries,!can);
   if (can) {
      if (TestBit(kVertDisplay)) fCandleAxis = new TGaxis(0.05,0.1,0.05,0.9,GetGlobalMin(),GetGlobalMax());
      else                       fCandleAxis = new TGaxis(0.1,0.05,0.9,0.05,GetGlobalMin(),GetGlobalMax());
      fCandleAxis->Draw();
   } else {
      if (fCandleAxis) {
         delete fCandleAxis;
         fCandleAxis = NULL;
      }
   }
   gPad->Modified();
   gPad->Update();
}


//______________________________________________________________________________
void TParallelCoord::SetCurrentFirst(Long64_t f)
{
   // Set the first entry to be dispayed.

   if(f<0 || f>fNentries) return;
   fCurrentFirst = f;
   if(fCurrentFirst + fCurrentN > fNentries) fCurrentN = fNentries-fCurrentFirst;
}


//______________________________________________________________________________
void TParallelCoord::SetCurrentN(Long64_t n)
{
   // Set the number of entry to be displayed.

   if(n<=0) return;
   if(fCurrentFirst+n>fNentries) fCurrentN = fNentries-fCurrentFirst;
   else fCurrentN = n;
}  


//______________________________________________________________________________
void TParallelCoord::SetCurrentSelection(const char* title)
{
   // Set the selection beeing edited.

   if (fCurrentSelection && fCurrentSelection->GetTitle() == title) return;
   TIter next(fSelectList);
   TParallelCoordSelect* sel;
   while((sel = (TParallelCoordSelect*)next()) && strcmp(sel->GetTitle(),title))
   if(!sel) return;
   fCurrentSelection = sel;
}


//______________________________________________________________________________
void TParallelCoord::SetCurrentSelection(TParallelCoordSelect* sel)
{
   // Set the selection beeing edited.

   if (fCurrentSelection == sel) return;
   fCurrentSelection = sel;
}


//______________________________________________________________________________
void TParallelCoord::SetDotsSpacing(Int_t s)
{
   // Set dots spacing. Modify the line style 11.

   if (s == fDotsSpacing) return;
   fDotsSpacing = s;
   gStyle->SetLineStyleString(11,Form("%d %d",4,s*8));
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalMax(Double_t max)
{
   // Force all variables to adopt the same max.

   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      var->SetCurrentMax(max);
   }
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalMin(Double_t min)
{
   // Force all variables to adopt the same min.

   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      var->SetCurrentMin(min);
   }
}


//______________________________________________________________________________
void TParallelCoord::SetLiveRangesUpdate(Bool_t on)
{
   // If true, the pad is updated while the motion of a dragged range.

   SetBit(kLiveUpdate,on);
   TIter next(fVarList);
   while(TParallelCoordVar* var = (TParallelCoordVar*)next()) var->SetLiveRangesUpdate(on);
}


//______________________________________________________________________________
void TParallelCoord::SetVertDisplay(Bool_t vert)
{
   // Set the vertical or horizontal display.

   if (vert == TestBit (kVertDisplay)) return;
   SetBit(kVertDisplay,vert);
   if (!gPad) return;
   UInt_t ui = 0;
   TFrame* frame = gPad->GetFrame();
   Double_t horaxisspace = (frame->GetX2() - frame->GetX1())/(fNvar-1);
   Double_t veraxisspace = (frame->GetY2() - frame->GetY1())/(fNvar-1);
   TIter next(fVarList);
   while (TParallelCoordVar* var = (TParallelCoordVar*)next()) {
      if (vert) var->SetX(frame->GetX1() + ui*horaxisspace,TestBit(kGlobalScale));
      else      var->SetY(frame->GetY1() + ui*veraxisspace,TestBit(kGlobalScale));
      ++ui;
   }
   if (TestBit(kCandleChart)) {
      if (fCandleAxis) delete fCandleAxis;
      if (TestBit(kVertDisplay)) fCandleAxis = new TGaxis(0.05,0.1,0.05,0.9,GetGlobalMin(),GetGlobalMax());
      else                       fCandleAxis = new TGaxis(0.1,0.05,0.9,0.05,GetGlobalMin(),GetGlobalMax());
      fCandleAxis->Draw();
   }
   gPad->Modified();
   gPad->Update();
}


//______________________________________________________________________________
void TParallelCoord::UnzoomAll()
{
   // Unzoom all variables.

   TIter next(fVarList);
   while(TParallelCoordVar* var = (TParallelCoordVar*)next()) var->Unzoom();
}
