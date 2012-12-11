// @(#)root/treeviewer:$Id$
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
#include "TFile.h"

ClassImp(TParallelCoord)

//______________________________________________________________________________
/* Begin_Html
<center><h2>Parallel Coordinates class</h2></center>
<p>
The multidimensional system of Parallel coordinates is a common way of studying high-dimensional geometry and visualizing multivariate problems. It has first been proposed by A. Inselberg in 1981.
<p>
To show a set of points in an n-dimensional space, a backdrop is drawn consisting of n parallel lines. A point in n-dimensional space is represented as a polyline with vertices on the parallel axes; the position of the vertex on the i-th axis corresponds to the i-th coordinate of the point.
<p>
This tool comes with a rather large gui in the editor. It is necessary to use this editor in order to explore a data set, as explained below.
<h4>Reduce cluttering:</h4>
<p>
The main issue for parallel coordinates is the very high cluttering of the output when dealing with large data set. Two techniques have been implemented to bypass that so far:
<ul>
<li>Draw doted lines instead of plain lines with an adjustable dots spacing. A slider to adjust the dots spacing is available in the editor.</li>
<li>Sort the entries to display with  a "weight cut". On each axis is drawn a histogram describing the distribution of the data on the corresponding variable. The "weight" of an entry is the sum of the bin content of each bin the entry is going through. An entry going through the histograms peaks will have a big weight wether an entry going randomly through the histograms will have a rather small weight. Setting a cut on this weight allows to draw only the most representative entries. A slider set the cut is also available in the gui.</li>
</ul>
<center><h2>Selections:</h2></center>
<p>
Selections of specific entries can be defined over the data se using parallel coordinates. With that representation, a selection is an ensemble of ranges defined on the axes. Ranges defined on the same axis are conjugated with OR (an entry must be in one or the other ranges to be selected). Ranges on different axes are are conjugated with AND (an entry must be in all the ranges to be selected). Several selections can be defined with different colors. It is possible to generate an entry list from a given selection and apply it to the tree using the editor ("Apply to tree" button).
<center><h2>Axes:</h2></center>
<p>
Options can be defined each axis separatly using the right mouse click. These options can be applied to every axes using the editor.
<ul>
<li>Axis width: If set to 0, the axis is simply a line. If higher, a color histogram is drawn on the axis.</li>
<li>Axis histogram height: If not 0, a usual bar histogram is drawn on the plot.</li>
</ul>
<p>
The order in which the variables are drawn is essential to see the clusters. The axes can be dragged to change their position. A zoom is also available. The logarithm scale is also available by right clicking on the axis.
<center><h2>Candle chart:</h2></center>
<p>
TParallelCoord can also be used to display a candle chart. In that mode, every variable is drawn in the same scale. The candle chart can be combined with the parallel coordinates mode, drawing the candle sticks over the axes.
End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1");
   TFile *f = TFile::Open("$ROOTSYS/tutorials/tree/cernstaff.root");
   TTree *T = (TTree*)f->Get("T");
   T->Draw("Age:Grade:Step:Cost:Division:Nation","","para");
   TParallelCoord* para = (TParallelCoord*)gPad->GetListOfPrimitives()->FindObject("ParaCoord");
   TParallelCoordVar* grade = (TParallelCoordVar*)para->GetVarList()->FindObject("Grade");
   grade->AddRange(new TParallelCoordRange(grade,11.5,14));
   para->AddSelection("less30");
   para->GetCurrentSelection()->SetLineColor(kViolet);
   TParallelCoordVar* age = (TParallelCoordVar*)para->GetVarList()->FindObject("Age");
   age->AddRange(new TParallelCoordRange(age,21,30));
   return c1;
}
End_Macro
Begin_Html
<p> Some references:
<ul>
<li> Alfred Inselberg's Homepage <http://www.math.tau.ac.il/~aiisreal>, with Visual Tutorial, History, Selected Publications and Applications.</li>
<li> A small, easy introduction <http://catt.okstate.edu/jones98/parallel.html> by Christopher V. Jones. </li>
<li> Almir Olivette Artero, Maria Cristina Ferreira de Oliveira, Haim Levkowitz, "Uncovering Clusters in Crowded Parallel Coordinates Visualizations," <i>infovis</i>, pp. 81-88,  IEEE Symposium on Information Visualization (INFOVIS'04),  2004. </li>
</ul>
End_Html
*/


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
   :TNamed("ParaCoord","ParaCoord")
{
   // Normal constructor, the datas must be added afterwards
   // with TParallelCoord::AddVariable(Double_t*,const char*).

   Init();
   Int_t estimate = tree->GetEstimate();
   if (nentries>estimate) {
      Warning("TParallelCoord","Call tree->SetEstimate(tree->GetEntries()) to display all the tree variables");
      fNentries = estimate;
   } else {
      fNentries = nentries;
   }
   fCurrentN = fNentries;
   fTree = tree;
   fTreeName = fTree->GetName();
   if (fTree->GetCurrentFile()) fTreeFileName = fTree->GetCurrentFile()->GetName();
   else fTreeFileName = "";
   fVarList = new TList();
   fSelectList = new TList();
   fCurrentSelection = new TParallelCoordSelect();
   fSelectList->Add(fCurrentSelection);
}


//______________________________________________________________________________
TParallelCoord::~TParallelCoord()
{
   // Destructor.

   if (fCurrentEntries) delete fCurrentEntries;
   if (fInitEntries != fCurrentEntries && fInitEntries != 0) delete fInitEntries;
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
   fCurrentSelection = sel;
}


//______________________________________________________________________________
void  TParallelCoord::ApplySelectionToTree()
{
   // Apply the current selection to the tree.

   if(!fTree) return;
   if(fSelectList) {
      if(fSelectList->GetSize() == 0) return;
      if(fCurrentSelection == 0) fCurrentSelection = (TParallelCoordSelect*)fSelectList->First();
   }
   fCurrentEntries = GetEntryList();
   fNentries = fCurrentEntries->GetN();
   fCurrentFirst = 0;
   fCurrentN = fNentries;
   fTree->SetEntryList(fCurrentEntries);
   TString varexp = "";
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) varexp.Append(Form(":%s",var->GetTitle()));
   varexp.Remove(TString::kLeading,':');
   TSelectorDraw* selector = (TSelectorDraw*)((TTreePlayer*)fTree->GetPlayer())->GetSelector();
   fTree->Draw(varexp.Data(),"","goff para");
   next.Reset();
   Int_t i = 0;
   while ((var = (TParallelCoordVar*)next())) {
      var->SetValues(fNentries, selector->GetVal(i));
      ++i;
   }
   if (fSelectList) {           // FIXME It would be better to update the selections by deleting
      fSelectList->Delete();    // the meaningless ranges (selecting everything or nothing for example)
      fCurrentSelection = 0;    // after applying a new entrylist to the tree.
   }
   gPad->Modified();
   gPad->Update();
}


//______________________________________________________________________________
void  TParallelCoord::BuildParallelCoord(TSelectorDraw* selector, Bool_t candle)
{
   // Call constructor and add the variables.

   TParallelCoord* pc = new TParallelCoord(selector->GetTree(),selector->GetNfill());
   pc->SetBit(kCanDelete);
   selector->SetObject(pc);
   TString varexp = "";
   for(Int_t i=0;i<selector->GetDimension();++i) {
      if (selector->GetVal(i)) {
         pc->AddVariable(selector->GetVal(i),selector->GetVar(i)->GetTitle());
         if (selector->GetVar(i)) varexp.Append(Form(":%s",selector->GetVar(i)->GetTitle()));
      }
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
   TParallelCoordSelect* select;
   while ((select = (TParallelCoordSelect*)next())){
      if(select->Contains(range)) select->Remove(range);
   }
}


//______________________________________________________________________________
void TParallelCoord::DeleteSelection(TParallelCoordSelect * sel)
{
   // Delete a selection.

   fSelectList->Remove(sel);
   delete sel;
   if(fSelectList->GetSize() == 0) fCurrentSelection = 0;
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

   if (!GetTree()) return;
   if (!fCurrentEntries) fCurrentEntries = fInitEntries;
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
      gPad->SetView(0);
   }
   gPad->Clear();
   if (!optcandle) {
      if (gPad && gPad->IsA() == TCanvas::Class()
         && !((TCanvas*)gPad)->GetShowEditor()) {
         ((TCanvas*)gPad)->ToggleEditor();
         ((TCanvas*)gPad)->ToggleEventStatus();
      }
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
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
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

   if (gPad && gPad->IsA() == TCanvas::Class())
      ((TCanvas*)gPad)->Selected(gPad,this,1);
}


//______________________________________________________________________________
void TParallelCoord::ExecuteEvent(Int_t /*entry*/, Int_t /*px*/, Int_t /*py*/)
{
   // Execute the corresponding entry.

   gPad->SetCursor(kHand);
}


//______________________________________________________________________________
TParallelCoordSelect* TParallelCoord::GetCurrentSelection()
{
   // Return the selection currently being edited.

   if (!fSelectList) return 0;
   if (!fCurrentSelection) {
      fCurrentSelection = (TParallelCoordSelect*)fSelectList->First();
   }
   return fCurrentSelection;
}


//______________________________________________________________________________
TEntryList* TParallelCoord::GetEntryList(Bool_t sel)
{
   // Get the whole entry list or one for a selection.

   if(!sel || fCurrentSelection->GetSize() == 0){ // If no selection is specified, return the entry list of all the entries.
      return fInitEntries;
   } else { // return the entry list corresponding to the current selection.
      TEntryList *enlist = new TEntryList(fTree);
      TIter next(fVarList);
      for (Long64_t li=0;li<fNentries;++li) {
         next.Reset();
         Bool_t inrange=kTRUE;
         TParallelCoordVar* var;
         while((var = (TParallelCoordVar*)next())){
            if(!var->Eval(li,fCurrentSelection)) inrange = kFALSE;
         }
         if(!inrange) continue;
         enlist->Enter(fCurrentEntries->GetEntry(li));
      }
      return enlist;
   }
}


//______________________________________________________________________________
Double_t TParallelCoord::GetGlobalMax()
{
   // return the global maximum.

   Double_t gmax=-DBL_MAX;
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
      if (gmax < var->GetCurrentMax()) gmax = var->GetCurrentMax();
   }
   return gmax;
}


//______________________________________________________________________________
Double_t TParallelCoord::GetGlobalMin()
{
   // return the global minimum.

   Double_t gmin=DBL_MAX;
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
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
TParallelCoordSelect* TParallelCoord::GetSelection(const char* title)
{
   // Get a selection from its title.

   TIter next(fSelectList);
   TParallelCoordSelect* sel;
   while ((sel = (TParallelCoordSelect*)next()) && strcmp(title,sel->GetTitle())) { }
   return sel;
}


//______________________________________________________________________________
TTree* TParallelCoord::GetTree()
{
   // return the tree if fTree is defined. If not, the method try to load the tree
   // from fTreeFileName.

   if (fTree) return fTree;
   if (fTreeFileName=="" || fTreeName=="") {
      Error("GetTree","Cannot load the tree: no tree defined!");
      return 0;
   }
   TFile *f = TFile::Open(fTreeFileName.Data());
   if (!f) {
      Error("GetTree","Tree file name : \"%s\" does not exsist (Are you in the correct directory?).",fTreeFileName.Data());
      return 0;
   } else if (f->IsZombie()) {
      Error("GetTree","while opening \"%s\".",fTreeFileName.Data());
      return 0;
   } else {
      fTree = (TTree*)f->Get(fTreeName.Data());
      if (!fTree) {
         Error("GetTree","\"%s\" not found in \"%s\".", fTreeName.Data(), fTreeFileName.Data());
         return 0;
      } else {
         fTree->SetEntryList(fCurrentEntries);
         TString varexp = "";
         TIter next(fVarList);
         TParallelCoordVar* var;
         while ((var = (TParallelCoordVar*)next())) varexp.Append(Form(":%s",var->GetTitle()));
         varexp.Remove(TString::kLeading,':');
         fTree->Draw(varexp.Data(),"","goff para");
         TSelectorDraw* selector = (TSelectorDraw*)((TTreePlayer*)fTree->GetPlayer())->GetSelector();
         next.Reset();
         Int_t i = 0;
         while ((var = (TParallelCoordVar*)next())) {
            var->SetValues(fNentries, selector->GetVal(i));
            ++i;
         }
         return fTree;
      }
   }
}


//______________________________________________________________________________
Double_t* TParallelCoord::GetVariable(const char* vartitle)
{
   // Get the variables values from its title.

   TIter next(fVarList);
   TParallelCoordVar* var = 0;
   while(((var = (TParallelCoordVar*)next()) != 0) && (var->GetTitle() != vartitle)) { }
   if(!var) return 0;
   else     return var->GetValues();
}


//______________________________________________________________________________
Double_t* TParallelCoord::GetVariable(Int_t i)
{
   // Get the variables values from its index.

   if(i<0 || (UInt_t)i>fNvar) return 0;
   else return ((TParallelCoordVar*)fVarList->At(i))->GetValues();
}


//______________________________________________________________________________
void TParallelCoord::Init()
{
   // Initialise the data members of TParallelCoord.

   fNentries = 0;
   fVarList = 0;
   fSelectList = 0;
   SetBit(kVertDisplay,kTRUE);
   SetBit(kCurveDisplay,kFALSE);
   SetBit(kPaintEntries,kTRUE);
   SetBit(kLiveUpdate,kFALSE);
   SetBit(kGlobalScale,kFALSE);
   SetBit(kCandleChart,kFALSE);
   SetBit(kGlobalLogScale,kFALSE);
   fTree = 0;
   fCurrentEntries = 0;
   fInitEntries = 0;
   fCurrentSelection = 0;
   fNvar = 0;
   fDotsSpacing = 0;
   fCurrentFirst = 0;
   fCurrentN = 0;
   fCandleAxis = 0;
   fWeightCut = 0;
   fLineWidth = 1;
   fLineColor = kGreen-8;
   fTreeName = "";
   fTreeFileName = "";
}

//______________________________________________________________________________
void TParallelCoord::Paint(Option_t* /*option*/)
{
   // Paint the parallel coordinates graph.

   if (!GetTree()) return;
   gPad->Range(0,0,1,1);
   TFrame *frame = gPad->GetFrame();
   frame->SetLineColor(gPad->GetFillColor());
   SetAxesPosition();
   if(TestBit(kPaintEntries)){
      PaintEntries(0);
      TIter next(fSelectList);
      TParallelCoordSelect* sel;
      while((sel = (TParallelCoordSelect*)next())) {
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

   TGraph    *gr     = 0;
   TPolyLine *pl     = 0;
   TAttLine  *evline = 0;

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
   TParallelCoordVar* var=0;
   while((var = (TParallelCoordVar*)next())) {
      if (!strcmp(var->GetTitle(),vartitle)) break;
   }
   if(!var) Error("RemoveVariable","\"%s\" not a variable",vartitle);
   fVarList->Remove(var);
   fNvar = fVarList->GetSize();
   SetAxesPosition();
   var->DeleteVariable();
   return var;
}


//______________________________________________________________________________
void TParallelCoord::ResetTree()
{
   // Reset the tree entry list to the initial one..

   if(!fTree) return;
   fTree->SetEntryList(fInitEntries);
   fCurrentEntries = fInitEntries;
   fNentries = fCurrentEntries->GetN();
   fCurrentFirst = 0;
   fCurrentN = fNentries;
   TString varexp = "";
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) varexp.Append(Form(":%s",var->GetTitle()));
   varexp.Remove(TString::kLeading,':');
   fTree->Draw(varexp.Data(),"","goff para");
   next.Reset();
   TSelectorDraw* selector = (TSelectorDraw*)((TTreePlayer*)fTree->GetPlayer())->GetSelector();
   Int_t i = 0;
   while ((var = (TParallelCoordVar*)next())) {
      var->SetValues(fNentries, selector->GetVal(i));
      ++i;
   }
   if (fSelectList) {           // FIXME It would be better to update the selections by deleting
      fSelectList->Delete();    // the meaningless ranges (selecting everything or nothing for example)
      fCurrentSelection = 0;    // after applying a new entrylist to the tree.
   }
   gPad->Modified();
   gPad->Update();
}


//______________________________________________________________________________
void TParallelCoord::SaveEntryLists(const char* filename, Bool_t overwrite)
{
   // Save the entry lists in a root file "filename.root".

   TString sfile = filename;
   if (sfile == "") sfile = Form("%s_parallelcoord_entries.root",fTree->GetName());

   TFile* f = TFile::Open(sfile.Data());
   if (f) {
      Warning("SaveEntryLists","%s already exists.", sfile.Data());
      if (!overwrite) return;
      else Warning("SaveEntryLists","Overwriting.");
      f = new TFile(sfile.Data(),"RECREATE");
   } else {
      f = new TFile(sfile.Data(),"CREATE");
   }
   gDirectory = f;
   fInitEntries->Write("initentries");
   fCurrentEntries->Write("currententries");
   Info("SaveEntryLists","File \"%s\" written.",sfile.Data());
}


//______________________________________________________________________________
void TParallelCoord::SavePrimitive(ostream & out, Option_t* options)
{
   // Save the TParallelCoord in a macro.

   TString opt = options;
   opt.ToLower();
   //Bool_t overwrite = opt.Contains("overwrite"); // Is there a way to specify "options" when saving ?
   // Save the entrylists.
   const char* filename = Form("%s_parallelcoord_entries.root",fTree->GetName());
   SaveEntryLists(filename,kTRUE); // FIXME overwriting by default.
   SaveTree(fTreeFileName,kTRUE);  // FIXME overwriting by default.
   out<<"   // Create a TParallelCoord."<<endl;
   out<<"   TFile *f = TFile::Open(\""<<fTreeFileName.Data()<<"\");"<<endl;
   out<<"   TTree* tree = (TTree*)f->Get(\""<<fTreeName.Data()<<"\");"<<endl;
   out<<"   TParallelCoord* para = new TParallelCoord(tree,"<<fNentries<<");"<<endl;
   out<<"   // Load the entrylists."<<endl;
   out<<"   TFile *entries = TFile::Open(\""<<filename<<"\");"<<endl;
   out<<"   TEntryList *currententries = (TEntryList*)entries->Get(\"currententries\");"<<endl;
   out<<"   tree->SetEntryList(currententries);"<<endl;
   out<<"   para->SetInitEntries((TEntryList*)entries->Get(\"initentries\"));"<<endl;
   out<<"   para->SetCurrentEntries(currententries);"<<endl;
   TIter next(fSelectList);
   TParallelCoordSelect* sel;
   out<<"   TParallelCoordSelect* sel;"<<endl;
   out<<"   para->GetSelectList()->Delete();"<<endl;
   while ((sel = (TParallelCoordSelect*)next())) {
      out<<"   para->AddSelection(\""<<sel->GetTitle()<<"\");"<<endl;
      out<<"   sel = (TParallelCoordSelect*)para->GetSelectList()->Last();"<<endl;
      out<<"   sel->SetLineColor("<<sel->GetLineColor()<<");"<<endl;
      out<<"   sel->SetLineWidth("<<sel->GetLineWidth()<<");"<<endl;
   }
   TIter nextbis(fVarList);
   TParallelCoordVar* var;
   TString varexp = "";
   while ((var = (TParallelCoordVar*)nextbis())) varexp.Append(Form(":%s",var->GetTitle()));
   varexp.Remove(TString::kLeading,':');
   out<<"   tree->Draw(\""<<varexp.Data()<<"\",\"\",\"goff para\");"<<endl;
   out<<"   TSelectorDraw* selector = (TSelectorDraw*)((TTreePlayer*)tree->GetPlayer())->GetSelector();"<<endl;
   nextbis.Reset();
   Int_t i=0;
   out<<"   TParallelCoordVar* var;"<<endl;
   while ((var = (TParallelCoordVar*)nextbis())) {
      out<<"   //***************************************"<<endl;
      out<<"   // Create the axis \""<<var->GetTitle()<<"\"."<<endl;
      out<<"   para->AddVariable(selector->GetVal("<<i<<"),\""<<var->GetTitle()<<"\");"<<endl;
      out<<"   var = (TParallelCoordVar*)para->GetVarList()->Last();"<<endl;
      var->SavePrimitive(out,"pcalled");
      ++i;
   }
   out<<"   //***************************************"<<endl;
   out<<"   // Set the TParallelCoord parameters."<<endl;
   out<<"   para->SetCurrentFirst("<<fCurrentFirst<<");"<<endl;
   out<<"   para->SetCurrentN("<<fCurrentN<<");"<<endl;
   out<<"   para->SetWeightCut("<<fWeightCut<<");"<<endl;
   out<<"   para->SetDotsSpacing("<<fDotsSpacing<<");"<<endl;
   out<<"   para->SetLineColor("<<GetLineColor()<<");"<<endl;
   out<<"   para->SetLineWidth("<<GetLineWidth()<<");"<<endl;
   out<<"   para->SetBit(TParallelCoord::kVertDisplay,"<<TestBit(kVertDisplay)<<");"<<endl;
   out<<"   para->SetBit(TParallelCoord::kCurveDisplay,"<<TestBit(kCurveDisplay)<<");"<<endl;
   out<<"   para->SetBit(TParallelCoord::kPaintEntries,"<<TestBit(kPaintEntries)<<");"<<endl;
   out<<"   para->SetBit(TParallelCoord::kLiveUpdate,"<<TestBit(kLiveUpdate)<<");"<<endl;
   out<<"   para->SetBit(TParallelCoord::kGlobalLogScale,"<<TestBit(kGlobalLogScale)<<");"<<endl;
   if (TestBit(kGlobalScale)) out<<"   para->SetGlobalScale(kTRUE);"<<endl;
   if (TestBit(kCandleChart)) out<<"   para->SetCandleChart(kTRUE);"<<endl;
   if (TestBit(kGlobalLogScale)) out<<"   para->SetGlobalLogScale(kTRUE);"<<endl;
   out<<endl<<"   para->Draw();"<<endl;
}


//______________________________________________________________________________
void TParallelCoord::SaveTree(const char* filename, Bool_t overwrite)
{
   // Save the tree in a file if fTreeFileName == "".

   if (!(fTreeFileName=="")) return;
   TString sfile = filename;
   if (sfile == "") sfile = Form("%s.root",fTree->GetName());

   TFile* f = TFile::Open(sfile.Data());
   if (f) {
      Warning("SaveTree","%s already exists.", sfile.Data());
      if (!overwrite) return;
      else Warning("SaveTree","Overwriting.");
      f = new TFile(sfile.Data(),"RECREATE");
   } else {
      f = new TFile(sfile.Data(),"CREATE");
   }
   gDirectory = f;
   fTree->Write(fTreeName.Data());
   fTreeFileName = sfile;
   Info("SaveTree","File \"%s\" written.",sfile.Data());
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

      TParallelCoordVar* var;
      while((var = (TParallelCoordVar*)next())){
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
   TParallelCoordVar *var;
   while((var = (TParallelCoordVar*)next())) var->SetHistogramBinning(n);
}


//______________________________________________________________________________
void TParallelCoord::SetAxisHistogramHeight(Double_t h)
{
   // Set the same histogram axis height for all axis.

   TIter next(fVarList);
   TParallelCoordVar *var;
   while((var = (TParallelCoordVar*)next())) var->SetHistogramHeight(h);
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalLogScale(Bool_t lt)
{
   // All axes in log scale.

   if (lt == TestBit(kGlobalLogScale)) return;
   SetBit(kGlobalLogScale,lt);
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) var->SetLogScale(lt);
   if (TestBit(kGlobalScale)) SetGlobalScale(kTRUE);
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalScale(Bool_t gl)
{
   // Constraint all axes to the same scale.

   SetBit(kGlobalScale,gl);
   if (fCandleAxis) {
      delete fCandleAxis;
      fCandleAxis = 0;
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
      TParallelCoordVar* var;
      while ((var = (TParallelCoordVar*)next())) var->GetHistogram();
   }
   gPad->Modified();
   gPad->Update();
}


//______________________________________________________________________________
void TParallelCoord::SetAxisHistogramLineWidth(Int_t lw)
{
   // Set the same histogram axis line width for all axis.

   TIter next(fVarList);
   TParallelCoordVar *var;
   while((var = (TParallelCoordVar*)next())) var->SetHistogramLineWidth(lw);
}


//______________________________________________________________________________
void TParallelCoord::SetCandleChart(Bool_t can)
{
   // Set a candle chart display.

   SetBit(kCandleChart,can);
   SetGlobalScale(can);
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
      var->SetBoxPlot(can);
      var->SetHistogramLineWidth(0);
   }
   if (fCandleAxis) delete fCandleAxis;
   fCandleAxis = 0;
   SetBit(kPaintEntries,!can);
   if (can) {
      if (TestBit(kVertDisplay)) fCandleAxis = new TGaxis(0.05,0.1,0.05,0.9,GetGlobalMin(),GetGlobalMax());
      else                       fCandleAxis = new TGaxis(0.1,0.05,0.9,0.05,GetGlobalMin(),GetGlobalMax());
      fCandleAxis->Draw();
   } else {
      if (fCandleAxis) {
         delete fCandleAxis;
         fCandleAxis = 0;
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
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
      var->GetMinMaxMean();
      var->GetHistogram();
      if (var->TestBit(TParallelCoordVar::kShowBox)) var->GetQuantiles();
   }
}


//______________________________________________________________________________
void TParallelCoord::SetCurrentN(Long64_t n)
{
   // Set the number of entry to be displayed.

   if(n<=0) return;
   if(fCurrentFirst+n>fNentries) fCurrentN = fNentries-fCurrentFirst;
   else fCurrentN = n;
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
      var->GetMinMaxMean();
      var->GetHistogram();
      if (var->TestBit(TParallelCoordVar::kShowBox)) var->GetQuantiles();
   }
}


//______________________________________________________________________________
TParallelCoordSelect* TParallelCoord::SetCurrentSelection(const char* title)
{
   // Set the selection beeing edited.

   if (fCurrentSelection && fCurrentSelection->GetTitle() == title) return fCurrentSelection;
   TIter next(fSelectList);
   TParallelCoordSelect* sel;
   while((sel = (TParallelCoordSelect*)next()) && strcmp(sel->GetTitle(),title))
   if (sel) fCurrentSelection = sel;
   return sel;
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
void TParallelCoord::SetEntryList(TParallelCoord* para, TEntryList* enlist)
{
   // Set the entry lists of "para".

   para->SetCurrentEntries(enlist);
   para->SetInitEntries(enlist);
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalMax(Double_t max)
{
   // Force all variables to adopt the same max.

   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
      var->SetCurrentMax(max);
   }
}


//______________________________________________________________________________
void TParallelCoord::SetGlobalMin(Double_t min)
{
   // Force all variables to adopt the same min.

   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
      var->SetCurrentMin(min);
   }
}


//______________________________________________________________________________
void TParallelCoord::SetLiveRangesUpdate(Bool_t on)
{
   // If true, the pad is updated while the motion of a dragged range.

   SetBit(kLiveUpdate,on);
   TIter next(fVarList);
   TParallelCoordVar* var;
   while((var = (TParallelCoordVar*)next())) var->SetLiveRangesUpdate(on);
}


//______________________________________________________________________________
void TParallelCoord::SetVertDisplay(Bool_t vert)
{
   // Set the vertical or horizontal display.

   if (vert == TestBit (kVertDisplay)) return;
   SetBit(kVertDisplay,vert);
   if (!gPad) return;
   TFrame* frame = gPad->GetFrame();
   if (!frame) return;
   UInt_t ui = 0;   
   Double_t horaxisspace = (frame->GetX2() - frame->GetX1())/(fNvar-1);
   Double_t veraxisspace = (frame->GetY2() - frame->GetY1())/(fNvar-1);
   TIter next(fVarList);
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) {
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
   TParallelCoordVar* var;
   while((var = (TParallelCoordVar*)next())) var->Unzoom();
}
