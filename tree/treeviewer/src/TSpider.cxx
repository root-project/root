// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  20/07/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSpider.h"
#include "TAttFill.h"
#include "TAttText.h"
#include "TAttLine.h"
#include "TGraphPolargram.h"
#include "TPolyLine.h"
#include "TNtuple.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TList.h"
#include "TSelectorDraw.h"
#include "TROOT.h"
#include "TEntryList.h"
#include "TLatex.h"
#include "TPad.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TArc.h"
#include "float.h"
#include "TEnv.h"

ClassImp(TSpider)

//______________________________________________________________________________
/* Begin_Html
<center><h2>Spider class</h2></center>
A spider view is a handy way to visualize a set of data stored in a TTree. It draws as
many polar axes as selected data members. For each of them, it draws on the axis
the position of the present event between the min and max of the data member.
Two modes are availables:
<ul>
<li> The spider view: With each points on the axes is drawn a polyline.</li>
<li> The segment view: For each data member is drawn an arc segment with the radius corresponding to the event.</li>
</ul>
The spider plot is available from the treeviewer called by "atree->StartViewer()", or simply by calling its constructor and defining the variables to display.
End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","TSpider example",200,10,700,700);
   TFile *f = new TFile("$(ROOTSYS)/tutorials/hsimple.root");
   if (!f || f->IsZombie()) {
      printf("Please run <ROOT location>/tutorials/hsimple.C before.");
      return;
   }
   TNtuple* ntuple = f->Get("ntuple");
   TString varexp = "px:py:pz:random:sin(px):log(px/py):log(pz)";
   TString select = "px>0 && py>0 && pz>0";
   TString options = "average";
   TSpider *spider = new TSpider(ntuple,varexp.Data(),select.Data(),options.Data());
   spider->Draw();
   c1->ToggleEditor();
   c1->Selected(c1,spider,1);
   return c1;
}
End_Macro
*/


//______________________________________________________________________________
TSpider::TSpider()
{
   // Default constructor.

   fDisplayAverage=kFALSE;
   fForceDim=kFALSE;
   fPolargram=NULL;
   fInput=NULL;
   fManager=NULL;
   fNcols=0;
   fNx=3;
   fNy=4;
   fPolyList=NULL;
   fSelect=NULL;
   fSelector=NULL;
   fTree=NULL;
   fMax=NULL;
   fMin=NULL;
   fAve=NULL;
   fCanvas=NULL;
   fAveragePoly=NULL;
   fEntry=0;
   fSuperposed=NULL;
   fShowRange=kFALSE;
   fAngularLabels=kFALSE;
   fAverageSlices=NULL;
   fSegmentDisplay=kFALSE;
   fNentries=0;
   fFirstEntry=0;
   fArraySize=0;
   fCurrentEntries = NULL;
   fFormulas = NULL;
}


//______________________________________________________________________________
TSpider::TSpider(TTree* tree ,const char *varexp, const char *selection,
                               Option_t *option, Long64_t nentries, Long64_t firstentry)
   : TAttFill(2,3003), TAttLine(1,1,1)
{
   // Normal constructor. Options are:

   UInt_t ui=0;

   fArraySize = 16;
   fTree=tree;
   fSelector= new TSelectorDraw();
   fFormulas= new TList();
   fInput= new TList();
   fInput->Add(new TNamed("varexp",""));
   fInput->Add(new TNamed("selection",""));
   fSelector->SetInputList(fInput);
   gROOT->GetListOfCleanups()->Add(this);
   fNx=2;
   fNy=2;
   fDisplayAverage=kFALSE;
   fSelect=NULL;
   fManager=NULL;
   fCanvas=NULL;
   fAveragePoly=NULL;
   fEntry=fFirstEntry;
   fSuperposed=NULL;
   fShowRange=kFALSE;
   fAngularLabels=kTRUE;
   fForceDim=kFALSE;
   fAverageSlices=NULL;
   fSegmentDisplay=kFALSE;
   if (firstentry < 0 || firstentry > tree->GetEstimate()) firstentry = 0;
   fFirstEntry = firstentry;
   if (nentries>0) fNentries = nentries;
   else fNentries = nentries = tree->GetEstimate()-firstentry;

   fEntry = fFirstEntry;

   fPolargram=NULL;
   fPolyList=NULL;

   fTree->SetScanField(fNx*fNy);
   fCurrentEntries = new Long64_t[fNx*fNy];
   for(ui=0;ui<fNx*fNy;++ui) fCurrentEntries[ui]=0;

   TString opt = option;

   if (opt.Contains("average")) fDisplayAverage=kTRUE;
   if (opt.Contains("showrange")) fShowRange=kTRUE;
   if (opt.Contains("segment")) fSegmentDisplay=kTRUE;

   fNcols=8;

   SetVariablesExpression(varexp);
   SetSelectionExpression(selection);
   SyncFormulas();
   InitVariables(firstentry,nentries);
}


//______________________________________________________________________________
TSpider::~TSpider()
{
   // Destructor.

   delete [] fCurrentEntries;
   if(fPolyList){
      fPolyList->Delete();
      delete fPolyList;
   }
   if(fAverageSlices)
      delete [] fAverageSlices;
   if(fFormulas){
      fFormulas->Delete();
      delete fFormulas;
   }
   if(fSelect) delete fSelect;
   if(fSelector) delete fSelector;
   if(fInput){
      fInput->Delete();
      delete fInput;
   }
   if(fMax) delete [] fMax;
   if(fMin) delete [] fMin;
   if(fAve) delete [] fAve;
   if(fSuperposed){
      fSuperposed->Delete();
      delete fSuperposed;
   }
   if (fCanvas) fCanvas->cd(0);
}


//______________________________________________________________________________
void TSpider::AddSuperposed(TSpider* sp)
{
   // Allow to superpose several spider views.

   if(!fSuperposed) fSuperposed=new TList();
   fSuperposed->Add(sp);
}


//______________________________________________________________________________
void TSpider::AddVariable(const char* varexp)
{
   // Add a variable to the plot from its expression.

   if(!strlen(varexp)) return;
   TTreeFormula *fvar = new TTreeFormula("Var1",varexp,fTree);
   if(fvar->GetNdim() <= 0) return;

   fFormulas->AddAfter(fFormulas->At(fNcols-1),fvar);

   InitArrays(fNcols + 1);
   ++fNcols;
   SyncFormulas();

   UInt_t ui=0;
   Long64_t notSkipped=0;
   Int_t tnumber=-1;
   Long64_t entryNumber;
   Long64_t entry = fFirstEntry;
   Int_t entriesToDisplay = fNentries;
   while(entriesToDisplay!=0){
      entryNumber = fTree->GetEntryNumber(entry);
      if(entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if(localEntry < 0) break;
      if(tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if(fManager) fManager->UpdateFormulaLeaves();
         else {
            for(Int_t i=0;i<=fFormulas->LastIndex();++i)
               ((TTreeFormula*)fFormulas->At(i))->UpdateFormulaLeaves();
         }
      }
      Int_t ndata=1;
      if(fForceDim){
         if(fManager)
            ndata = fManager->GetNdata(kTRUE);
         else {
            for(ui=0;ui<fNcols;++ui){
               if(ndata<((TTreeFormula*)fFormulas->At(ui))->GetNdata())
                  ndata = ((TTreeFormula*)fFormulas->At(ui))->GetNdata();
            }
            if(fSelect && fSelect->GetNdata() == 0)
               ndata = 0;
         }
      }

      Bool_t loaded = kFALSE;
      Bool_t skip = kFALSE;
      // Loop over the instances of the selection condition
      for(Int_t inst=0;inst<ndata;++inst){
         if(fSelect){
            if(fSelect->EvalInstance(inst) == 0){
               skip = kTRUE;
               ++entry;
            }
         }
         if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            ((TTreeFormula*)fFormulas->At(fNcols-1))->EvalInstance(0);
            loaded = kTRUE;
         } else if (inst == 0) {
            loaded = kTRUE;
         }
      }
      if(!skip){
         fTree->LoadTree(entryNumber);
         TTreeFormula* var = (TTreeFormula*)fFormulas->At(fNcols-1);
         if(var->EvalInstance()>fMax[fNcols-1]) fMax[fNcols-1]=var->EvalInstance();
         if(var->EvalInstance()<fMin[fNcols-1]) fMin[fNcols-1]=var->EvalInstance();
         fAve[fNcols-1]+=var->EvalInstance();
         ++notSkipped;
         --entriesToDisplay;
         ++entry;
      }
   }
   fAve[fNcols-1]/=notSkipped;

   Color_t lc;
   Style_t lt;
   Width_t lw;
   Color_t fc;
   Style_t fs;

   if(fAverageSlices){
      lc = fAverageSlices[0]->GetLineColor();
      lt = fAverageSlices[0]->GetLineStyle();
      lw = fAverageSlices[0]->GetLineWidth();
      fc = fAverageSlices[0]->GetFillColor();
      fs = fAverageSlices[0]->GetFillStyle();
   } else {
      lc = fAveragePoly->GetLineColor();
      lt = fAveragePoly->GetLineStyle();
      lw = fAveragePoly->GetLineWidth();
      fc = fAveragePoly->GetFillColor();
      fs = fAveragePoly->GetFillStyle();
   }

   delete fPolargram;
   fPolargram = NULL;

   if(fSegmentDisplay){
      for(ui=0;ui<fNx*fNy;++ui) ((TList*)fPolyList->At(ui))->Delete();
      if (fAverageSlices) for(ui=0;ui<fNcols-1;++ui) delete fAverageSlices[ui];
   }
   fPolyList->Delete();
   delete fPolyList;
   fPolyList = NULL;
   delete [] fAverageSlices;
   fAverageSlices = NULL;
   delete fAveragePoly;
   fAveragePoly = NULL;

   if (fCanvas) {
      fCanvas->Clear();
      fCanvas->Divide(fNx,fNy);
   }
   Draw("");

   if(fAverageSlices){
      for(ui = 0;ui<fNcols;++ui){
         fAverageSlices[ui]->SetLineColor(lc);
         fAverageSlices[ui]->SetLineStyle(lt);
         fAverageSlices[ui]->SetLineWidth(lw);
         fAverageSlices[ui]->SetFillColor(fc);
         fAverageSlices[ui]->SetFillStyle(fs);
      }
   } else {
      fAveragePoly->SetLineColor(lc);
      fAveragePoly->SetLineStyle(lt);
      fAveragePoly->SetLineWidth(lw);
      fAveragePoly->SetFillColor(fc);
      fAveragePoly->SetFillStyle(fs);
   }
}


//______________________________________________________________________________
void TSpider::DeleteVariable(const char* varexp)
{
   // Delete a variable from its expression.

   Int_t var=-1;
   UInt_t ui=0;

   if(fNcols == 2) return;
   for(ui=0; ui<fNcols;++ui){
      if(!strcmp(varexp,((TTreeFormula*)fFormulas->At(ui))->GetTitle())) var = ui;
   }
   if(var<0) return;

   fFormulas->Remove(fFormulas->At(var));
   SyncFormulas();

   for(ui=var+1;ui<fNcols;++ui){
      fMin[ui-1] = fMin[ui];
      fMax[ui-1] = fMax[ui];
      fAve[ui-1] = fAve[ui];
   }
   fMin[fNcols-1] = FLT_MAX;
   fMax[fNcols-1] = -FLT_MAX;
   fAve[fNcols-1] = 0;
   --fNcols;

   Color_t lc;
   Style_t lt;
   Width_t lw;
   Color_t fc;
   Style_t fs;

   if(fAverageSlices){
      lc = fAverageSlices[0]->GetLineColor();
      lt = fAverageSlices[0]->GetLineStyle();
      lw = fAverageSlices[0]->GetLineWidth();
      fc = fAverageSlices[0]->GetFillColor();
      fs = fAverageSlices[0]->GetFillStyle();
   } else {
      lc = fAveragePoly->GetLineColor();
      lt = fAveragePoly->GetLineStyle();
      lw = fAveragePoly->GetLineWidth();
      fc = fAveragePoly->GetFillColor();
      fs = fAveragePoly->GetFillStyle();
   }

   delete fPolargram;
   fPolargram = NULL;

   if(fSegmentDisplay){
      for(ui=0;ui<fNx*fNy;++ui) ((TList*)fPolyList->At(ui))->Delete();
      if (fAverageSlices) for(ui=0;ui<=fNcols;++ui) delete fAverageSlices[ui];
   }
   fPolyList->Delete();
   delete fPolyList;
   fPolyList = NULL;
   delete [] fAverageSlices;
   fAverageSlices = NULL;
   delete fAveragePoly;
   fAveragePoly = NULL;

   if (fCanvas) {
      fCanvas->Clear();
      fCanvas->Divide(fNx,fNy);
   }
   Draw("");
   if(fNcols == 2) SetSegmentDisplay(kTRUE);

   if(fAverageSlices){
      for(ui = 0;ui<fNcols;++ui){
         fAverageSlices[ui]->SetLineColor(lc);
         fAverageSlices[ui]->SetLineStyle(lt);
         fAverageSlices[ui]->SetLineWidth(lw);
         fAverageSlices[ui]->SetFillColor(fc);
         fAverageSlices[ui]->SetFillStyle(fs);
      }
   } else {
      fAveragePoly->SetLineColor(lc);
      fAveragePoly->SetLineStyle(lt);
      fAveragePoly->SetLineWidth(lw);
      fAveragePoly->SetFillColor(fc);
      fAveragePoly->SetFillStyle(fs);
   }
}


//______________________________________________________________________________
Int_t TSpider::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute the distance to the spider.

   if(!gPad) return 9999;
   Double_t xx,yy,r2;
   xx=gPad->AbsPixeltoX(px);
   yy=gPad->AbsPixeltoY(py);
   r2 = xx*xx + yy*yy;
   if(r2>1 && r2<1.5)
      return 0;
   else return 9999;
}


//______________________________________________________________________________
void TSpider::Draw(Option_t *options)
{
   // Draw the spider.

   UInt_t ui=0;

   gEnv->SetValue("Canvas.ShowEditor",1);
   if(!gPad && !fCanvas){
      fCanvas = new TCanvas("screen","Spider Plot",fNx*256,fNy*256);
      if (fCanvas) fCanvas->Divide(fNx,fNy);
   } else if(!fCanvas){
      fCanvas = (TCanvas*)gPad;
      if (fCanvas) fCanvas->Divide(fNx,fNy);
   }
   if(fPolargram) delete fPolargram;
   fPolargram=new TGraphPolargram("fPolargram");
   fPolargram->SetNdivPolar(fNcols);
   fPolargram->SetNdivRadial(0);
   if (fCanvas) fCanvas->cd();
   SetCurrentEntries();
   AppendPad(options);
   for(ui=0;ui<fNx*fNy;++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      fPolargram->Draw("pn");
      fTree->LoadTree(fCurrentEntries[ui]);
      if(fSegmentDisplay){
         if(fDisplayAverage) DrawSlicesAverage("");
         DrawSlices("");
      } else {
         if(fDisplayAverage) DrawPolyAverage("");
         DrawPoly("");
      }
      AppendPad();
   }
   if (fCanvas) fCanvas->Selected(fCanvas,this,1);
}


//______________________________________________________________________________
void TSpider::DrawPolyAverage(Option_t* /*options*/)
{
   // Paint the Polygon representing the average value of the variables.

   Int_t linecolor=4;
   Int_t fillstyle=0;
   Int_t fillcolor=linecolor;
   Int_t linewidth=1;
   Int_t linestyle=1;

   UInt_t ui=0;
   Double_t slice = 2*TMath::Pi()/fNcols;
   Double_t *x = new Double_t[fNcols+1];
   Double_t *y = new Double_t[fNcols+1];

   for(ui=0;ui<fNcols;++ui){
      x[ui]=(fAve[ui]-fMin[ui])/(fMax[ui]-fMin[ui])*TMath::Cos(ui*slice);
      y[ui]=(fAve[ui]-fMin[ui])/(fMax[ui]-fMin[ui])*TMath::Sin(ui*slice);
   }
   x[fNcols]=(fAve[0]-fMin[0])/(fMax[0]-fMin[0]);
   y[fNcols]=0;

   if(!fAveragePoly){
      fAveragePoly = new TPolyLine(fNcols+1,x,y);
      fAveragePoly->SetLineColor(linecolor);
      fAveragePoly->SetLineWidth(linewidth);
      fAveragePoly->SetLineStyle(linestyle);
      fAveragePoly->SetFillStyle(fillstyle);
      fAveragePoly->SetFillColor(fillcolor);
   }
   fAveragePoly->Draw();
   fAveragePoly->Draw("f");

   delete [] x;
   delete [] y;
}


//______________________________________________________________________________
void TSpider::DrawPoly(Option_t* /*options*/)
{
   // Paint the polygon representing the current entry.

   if(!fPolyList) fPolyList = new TList();
   Double_t *x = new Double_t[fNcols+1];
   Double_t *y = new Double_t[fNcols+1];

   Double_t slice = 2*TMath::Pi()/fNcols;
   for(UInt_t i=0;i<fNcols;++i){
      x[i]=(((TTreeFormula*)fFormulas->At(i))->EvalInstance()-fMin[i])/(fMax[i]-fMin[i])*TMath::Cos(i*slice);
      y[i]=(((TTreeFormula*)fFormulas->At(i))->EvalInstance()-fMin[i])/(fMax[i]-fMin[i])*TMath::Sin(i*slice);
   }
   x[fNcols]=(((TTreeFormula*)fFormulas->At(0))->EvalInstance()-fMin[0])/(fMax[0]-fMin[0]);
   y[fNcols]=0;

   TPolyLine* poly= new TPolyLine(fNcols+1,x,y);
   poly->SetFillColor(GetFillColor());
   poly->SetFillStyle(GetFillStyle());
   poly->SetLineWidth(GetLineWidth());
   poly->SetLineColor(GetLineColor());
   poly->SetLineStyle(GetLineStyle());
   poly->Draw("f");
   poly->Draw();
   fPolyList->Add(poly);
   delete [] x;
   delete [] y;
}


//______________________________________________________________________________
void TSpider::DrawSlices(Option_t* options)
{
   // Draw the slices of the segment plot.

   UInt_t ui=0;

   Double_t angle = 2*TMath::Pi()/fNcols;
   Double_t conv = 180.0/TMath::Pi();

   if(!fPolyList) fPolyList = new TList;
   TList* li = new TList();
   for(ui=0;ui<fNcols;++ui){
      Double_t r = (((TTreeFormula*)fFormulas->At(ui))->EvalInstance()-fMin[ui])/(fMax[ui]-fMin[ui]);
      TArc* slice = new TArc(0,0,r,(ui-0.25)*angle*conv,(ui+0.25)*angle*conv);
      slice->SetFillColor(GetFillColor());
      slice->SetFillStyle(GetFillStyle());
      slice->SetLineWidth(GetLineWidth());
      slice->SetLineColor(GetLineColor());
      slice->SetLineStyle(GetLineStyle());
      li->Add(slice);
      slice->Draw(options);
   }
   fPolyList->Add(li);
}


//______________________________________________________________________________
void TSpider::DrawSlicesAverage(Option_t* /*options*/)
{
   // Draw the slices representing the average for the segment plot.

   UInt_t ui=0;

   Int_t fillstyle=3002;
   Int_t linecolor=4;
   Int_t fillcolor=linecolor;
   Int_t linewidth=1;
   Int_t linestyle=1;

   Double_t angle = 2*TMath::Pi()/fNcols;
   Double_t conv = 180.0/TMath::Pi();

   if(!fAverageSlices){
      fAverageSlices = new TArc*[fNcols];
      for(ui=0;ui<fNcols;++ui){
         Double_t r = (fAve[ui]-fMin[ui])/(fMax[ui]-fMin[ui]);
         fAverageSlices[ui] = new TArc(0,0,r,(ui-0.5)*angle*conv,(ui+0.5)*angle*conv);
         fAverageSlices[ui]->SetFillColor(fillcolor);
         fAverageSlices[ui]->SetFillStyle(fillstyle);
         fAverageSlices[ui]->SetLineWidth(linewidth);
         fAverageSlices[ui]->SetLineColor(linecolor);
         fAverageSlices[ui]->SetLineStyle(linestyle);
      }
   }
   for(ui=0;ui<fNcols;++ui) fAverageSlices[ui]->Draw();
}


//______________________________________________________________________________
Style_t TSpider::GetAverageLineStyle() const
{
   // Get the LineStyle of the average.

   if(fAverageSlices) return fAverageSlices[0]->GetLineStyle();
   else if(fAveragePoly) return fAveragePoly->GetLineStyle();
   else return 0;
}


//______________________________________________________________________________
Color_t TSpider::GetAverageLineColor() const
{
   // Get the LineColor of the average.

   if(fAverageSlices) return fAverageSlices[0]->GetLineColor();
   else if(fAveragePoly) return fAveragePoly->GetLineColor();
   else return 0;
}


//______________________________________________________________________________
Width_t TSpider::GetAverageLineWidth() const
{
   // Get the LineWidth of the average.

   if(fAverageSlices) return fAverageSlices[0]->GetLineWidth();
   else if(fAveragePoly) return fAveragePoly->GetLineWidth();
   else return 0;
}


//______________________________________________________________________________
Color_t TSpider::GetAverageFillColor() const
{
   // Get the FillColor of the average.

   if(fAverageSlices) return fAverageSlices[0]->GetFillColor();
   else if(fAveragePoly) return fAveragePoly->GetFillColor();
   else return 0;
}


//______________________________________________________________________________
Style_t TSpider::GetAverageFillStyle() const
{
   // Get the FillStyle of the average.

   if(fAverageSlices) return fAverageSlices[0]->GetFillStyle();
   else if(fAveragePoly) return fAveragePoly->GetFillStyle();
   else return 0;
}


//______________________________________________________________________________
void TSpider::ExecuteEvent(Int_t /*event*/,Int_t /*px*/, Int_t /*py*/)
{
   // Execute the corresponding event.

   gPad->SetCursor(kHand);
}


//______________________________________________________________________________
Int_t TSpider::FindTextAlign(Double_t angle)
{
   // Find the alignement rule to apply for TText::SetTextAlign(Short_t).

   Double_t pi = TMath::Pi();

   while(angle < 0 || angle > 2*pi){
      if(angle < 0) angle+=2*pi;
      if(angle > 2*pi) angle-=2*pi;
   }
   if(!fAngularLabels){
      if(angle > 0 && angle < pi/2) return 11;
      else if(angle > pi/2 && angle < pi) return 31;
      else if(angle > pi && angle < 3*pi/2) return 33;
      else if(angle > 3*pi/2 && angle < 2*pi) return 13;
      else if(angle == 0 || angle == 2*pi) return 12;
      else if(angle == pi/2) return 21;
      else if(angle == pi) return 32;
      else if(angle == 3*pi/2) return 23;
      else return 0;
   }
   else{
      if(angle >= 0 && angle < pi) return 21;
      else if(angle >=pi && angle <= 2*pi) return 23;
      else return 0;
   }
}


//______________________________________________________________________________
Double_t TSpider::FindTextAngle(Double_t angle)
{
   // Determine the orientation of the polar labels according to their angle.

   Double_t pi = TMath::Pi();
   Double_t convraddeg = 180.0/pi;

   while(angle < 0 || angle > 2*pi){
      if(angle < 0) angle+=2*pi;
      if(angle > 2*pi) angle-=2*pi;
   }

   if(angle >= 0 && angle <= pi/2) return angle*convraddeg - 90;
   else if(angle > pi/2 && angle <= pi) return (angle + pi)*convraddeg + 90;
   else if(angle > pi && angle <= 3*pi/2) return (angle - pi)*convraddeg - 90;
   else if(angle > 3*pi/2 && angle <= 2*pi) return angle*convraddeg + 90;
   else return 0;
}


//______________________________________________________________________________
Long64_t TSpider::GetEntriesToProcess(Long64_t firstentry, Long64_t nentries) const
{
   // return the number of entries to be processed
   // this function checks that nentries is not bigger than the number
   // of entries in the Tree or in the associated TEventlist

   Long64_t lastentry = firstentry + nentries - 1;
   if (lastentry > fTree->GetEntriesFriend()-1) {
      lastentry  = fTree->GetEntriesFriend() - 1;
      nentries   = lastentry - firstentry + 1;
   }
   //TEventList *elist = fTree->GetEventList();
   //if (elist && elist->GetN() < nentries) nentries = elist->GetN();
   TEntryList *elist = fTree->GetEntryList();
   if (elist && elist->GetN() < nentries) nentries = elist->GetN();
   return nentries;
}


//______________________________________________________________________________
void TSpider::GotoEntry(Long64_t e)
{
   // Go to a specified entry.

   if(e<fFirstEntry || e+fTree->GetScanField()>=fFirstEntry + fNentries) return;
   fEntry = e;
   SetCurrentEntries();
}


//______________________________________________________________________________
void TSpider::GotoNext()
{
   // Go to the next entries.

   if(fEntry + 2*fTree->GetScanField() -1 >= fFirstEntry + fNentries) fEntry = fFirstEntry;
   else fEntry=fCurrentEntries[fTree->GetScanField()-1]+1;
   SetCurrentEntries();
}


//______________________________________________________________________________
void TSpider::GotoPrevious()
{
   // Go to the prevous entries.

   if(fEntry-fTree->GetScanField() < fFirstEntry) fEntry = fFirstEntry + fNentries -1 - fTree->GetScanField();
   else fEntry -= fTree->GetScanField();
   SetCurrentEntries();
}


//______________________________________________________________________________
void TSpider::GotoFollowing()
{
   // Go to the next entry.

   if(fEntry + fTree->GetScanField() >= fFirstEntry + fNentries) return;
   ++fEntry;
   SetCurrentEntries();
}


//______________________________________________________________________________
void TSpider::GotoPreceding()
{
   // Go to the last entry.

   if(fEntry - 1 < fFirstEntry) return;
   --fEntry;
   SetCurrentEntries();
}


//______________________________________________________________________________
void TSpider::InitArrays(Int_t newsize)
{
   // Check if the arrays size is enough and reallocate them if not.

   if(newsize>fArraySize){

      Int_t i;
      Int_t old = fArraySize;

      while(fArraySize<newsize) fArraySize*=2;

      Double_t *memmax = new Double_t[fArraySize];
      Double_t *memmin = new Double_t[fArraySize];
      Double_t *memave = new Double_t[fArraySize];

      for(i=0;i<fArraySize;++i){
         if(i<old){
            memmax[i] = fMax[i];
            memmin[i] = fMin[i];
            memave[i] = fAve[i];
         } else {
            memmax[i] = -FLT_MAX;
            memmin[i] = FLT_MAX;
            memave[i] = 0;
         }
      }

      delete [] fMax;
      delete [] fMin;
      delete [] fAve;

      fMax = memmax;
      fMin = memmin;
      fAve = memave;
   }
}


//______________________________________________________________________________
void TSpider::InitVariables(Long64_t firstentry, Long64_t nentries)
{
   // Browse the tree to set the min, max and average value of each variable of fVar.

   UInt_t ui=0;
   Int_t i;

   fMax = new Double_t [fArraySize];
   fMin= new Double_t [fArraySize];
   fAve= new Double_t [fArraySize];

   for(i=0;i<fArraySize;++i){
      fMax[i]= -FLT_MAX;
      fMin[i]= FLT_MAX;
      fAve[i]=0;
   }

   Long64_t notSkipped=0;
   Int_t tnumber=-1;
   Long64_t entryNumber;
   Long64_t entry = firstentry;
   Int_t entriesToDisplay = nentries;
   while(entriesToDisplay!=0){
      entryNumber = fTree->GetEntryNumber(entry);
      if(entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if(localEntry < 0) break;
      if(tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if(fManager) fManager->UpdateFormulaLeaves();
         else {
            for(i=0;i<=fFormulas->LastIndex();++i)
               ((TTreeFormula*)fFormulas->At(i))->UpdateFormulaLeaves();
         }
      }
      Int_t ndata=1;
      if(fForceDim){
         if(fManager)
            ndata = fManager->GetNdata(kTRUE);
         else {
            for(ui=0;ui<fNcols;++ui){
               if(ndata<((TTreeFormula*)fFormulas->At(ui))->GetNdata())
                  ndata = ((TTreeFormula*)fFormulas->At(ui))->GetNdata();
            }
            if(fSelect && fSelect->GetNdata() == 0)
               ndata = 0;
         }
      }
      Bool_t loaded = kFALSE;
      Bool_t skip = kFALSE;
      // Loop over the instances of the selection condition
      for(Int_t inst=0;inst<ndata;++inst){
         if(fSelect){
            if(fSelect->EvalInstance(inst) == 0){
               skip = kTRUE;
               ++entry;
            }
         }
         if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            for (ui=0;ui<fNcols;ui++) {
               ((TTreeFormula*)fFormulas->At(ui))->EvalInstance(0);
            }
            loaded = kTRUE;
         } else if (inst == 0) {
            loaded = kTRUE;
         }
      }
      if(!skip){
         fTree->LoadTree(entryNumber);
         for(ui=0;ui<fNcols;++ui){
            Double_t inst = ((TTreeFormula*)fFormulas->At(ui))->EvalInstance();
            if(inst > fMax[ui]) fMax[ui] = inst;
            if(inst < fMin[ui]) fMin[ui] = inst;
            fAve[ui] += inst;
         }
         ++notSkipped;
         --entriesToDisplay;
         ++entry;
      }
   }
   for(ui=0;ui<fNcols;++ui) fAve[ui]/=notSkipped;
}


//______________________________________________________________________________
void TSpider::Paint(Option_t* options)
{
   // Paint the spider.

   UInt_t ui=0;
   TString opt = options;

   if(opt.Contains("n")) return;

   Double_t slice = 2*TMath::Pi()/fNcols;
   Double_t offset(1.0);
   if (!fCanvas) {
      if (gPad) fCanvas = gPad->GetCanvas();
      else return;
   }

   TLatex *txt = new TLatex();
   for(ui=0;ui<fNx*fNy;++ui){
      txt->SetTextAlign(13);
      if (fCanvas) fCanvas->cd(ui+1);
      if (fCurrentEntries) {
         txt->PaintLatex(-1.2,1.2,0,0.08,Form("#%d",(int)fCurrentEntries[ui]));
      }
      txt->SetTextSize(0.035);
      for(UInt_t var=0;var<fNcols;++var){ // Print labels.
         if(ui==0){
            txt->SetTextAlign(FindTextAlign(var*slice));
            offset = 1.09 + txt->GetTextSize();
            txt->PaintLatex(offset*TMath::Cos(var*slice),offset*TMath::Sin(var*slice),
                            FindTextAngle(var*slice),0.035,fFormulas->At(var)->GetTitle());
            offset= 1.03;
            txt->PaintLatex(offset*TMath::Cos(var*slice),offset*TMath::Sin(var*slice),
                            FindTextAngle(var*slice),0.035,Form("[%5.3f,%5.3f]",fMin[var],fMax[var]));
         }
         else {
            txt->SetTextAlign(FindTextAlign(var*slice));
            if(var*slice >=0 && var*slice <= TMath::Pi()) offset =1.13 + txt->GetTextSize();
            else offset = 1.09 + txt->GetTextSize();
            txt->PaintLatex(offset*TMath::Cos(var*slice),offset*TMath::Sin(var*slice),
                            FindTextAngle(var*slice),0.035,fFormulas->At(var)->GetTitle());
         }
      }
   }
   delete txt;
}


//______________________________________________________________________________
void TSpider::SetAverageLineStyle(Style_t sty)
{
   // Set the LineStyle of the average.

   UInt_t ui=0;

   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui) fAverageSlices[ui]->SetLineStyle(sty);
   } else if(fAveragePoly) fAveragePoly->SetLineStyle(sty);
}


//______________________________________________________________________________
void TSpider::SetAverageLineColor(Color_t col)
{
   // Set the LineColor of the average.

   UInt_t ui=0;

   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui) fAverageSlices[ui]->SetLineColor(col);
   } else if(fAveragePoly) fAveragePoly->SetLineColor(col);
}


//______________________________________________________________________________
void TSpider::SetAverageLineWidth(Width_t wid)
{
   // Set the LineWidth of the average.

   UInt_t ui=0;

   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui) fAverageSlices[ui]->SetLineWidth(wid);
   } else if(fAveragePoly) fAveragePoly->SetLineWidth(wid);
}


//______________________________________________________________________________
void TSpider::SetAverageFillColor(Color_t col)
{
   // Set the FillColor of the average.

   UInt_t ui=0;

   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui) fAverageSlices[ui]->SetFillColor(col);
   } else if(fAveragePoly) fAveragePoly->SetFillColor(col);
}


//______________________________________________________________________________
void TSpider::SetAverageFillStyle(Style_t sty)
{
   // Set the FillStyle of the average.

   UInt_t ui=0;

   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui) fAverageSlices[ui]->SetFillStyle(sty);
   } else if(fAveragePoly) fAveragePoly->SetFillStyle(sty);
}


//______________________________________________________________________________
void TSpider::SetDisplayAverage(Bool_t disp)
{
   // Display or not the average.

   if(disp == fDisplayAverage) return;

   UInt_t ui=0;

   fDisplayAverage = disp;
   delete fAveragePoly;
   fAveragePoly = NULL;
   if(fAverageSlices){
      for(ui = 0;ui<fNcols;++ui) delete fAverageSlices[ui];
   }
   delete [] fAverageSlices;
   fAverageSlices = NULL;

   for(ui=0;ui<fNx*fNy;++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      gPad->Clear();
   }

   for(ui = 0; ui < fNx*fNy; ++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      fPolargram->Draw("pn");
      fTree->LoadTree(fCurrentEntries[ui]);
      if(fSegmentDisplay){
         if(disp) DrawSlicesAverage("");
         DrawSlices("");
      } else {
         if(disp) DrawPolyAverage("");
         DrawPoly("");
      }
      AppendPad();
   }
   if (fCanvas) {
      fCanvas->Modified();
      fCanvas->Update();
   }
}


//______________________________________________________________________________
void TSpider::SetCurrentEntries()
{
   // Set the current selected entries.

   Int_t i;
   UInt_t ui=0;
   Int_t tnumber=-1;
   Long64_t entryNumber;
   Long64_t entry = fEntry;
   Int_t entriesToDisplay = fTree->GetScanField();

   if(!fCurrentEntries) fCurrentEntries = new Long64_t[fTree->GetScanField()];

   while(entriesToDisplay!=0){
      entryNumber = fTree->GetEntryNumber(entry);
      if(entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if(localEntry < 0) break;
      if(tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if(fManager) fManager->UpdateFormulaLeaves();
         else {
            for(i=0;i<=fFormulas->LastIndex();++i)
               ((TTreeFormula*)fFormulas->At(i))->UpdateFormulaLeaves();
         }
      }
      Int_t ndata=1;
      if(fForceDim){
         if(fManager)
            ndata = fManager->GetNdata(kTRUE);
         else {
            for(ui=0;ui<fNcols;++ui){
               if(ndata < ((TTreeFormula*)fFormulas->At(ui))->GetNdata())
                  ndata = ((TTreeFormula*)fFormulas->At(ui))->GetNdata();
            }
            if(fSelect && fSelect->GetNdata() == 0)
               ndata = 0;
         }
      }
      Bool_t loaded = kFALSE;
      Bool_t skip = kFALSE;
      // Loop over the instances of the selection condition
      for(Int_t inst=0;inst<ndata;++inst){
         if(fSelect){
            if(fSelect->EvalInstance(inst) == 0){
               skip = kTRUE;
               ++entry;
            }
         }
         if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            for (ui=0;ui<fNcols;ui++) {
               ((TTreeFormula*)fFormulas->At(ui))->EvalInstance(0);
            }
            loaded = kTRUE;
         } else if (inst == 0) {
            loaded = kTRUE;
         }
      }
      if(!skip){
          fCurrentEntries[fTree->GetScanField()-entriesToDisplay] = entryNumber;
         --entriesToDisplay;
         ++entry;
      }
   }
   if(fPolyList) UpdateView();
}


//______________________________________________________________________________
void TSpider::SetLineStyle(Style_t sty)
{
   // Set line style.

   UInt_t ui=0;

   TAttLine::SetLineStyle(sty);
   for(ui=0; ui<fNx*fNy;++ui){
      if(fSegmentDisplay){
         TList *li = (TList*)fPolyList->At(ui);
         for(UInt_t var=0;var<fNcols;++var) ((TArc*)li->At(var))->SetLineStyle(sty);
      } else ((TPolyLine*)fPolyList->At(ui))->SetLineStyle(sty);
   }
}


//______________________________________________________________________________
void TSpider::SetLineColor(Color_t col)
{
   // Set lin color.

   UInt_t ui=0;

   TAttLine::SetLineColor(col);
   for(ui=0; ui<fNx*fNy;++ui){
      if(fSegmentDisplay){
         TList *li = (TList*)fPolyList->At(ui);
         for(UInt_t var=0;var<fNcols;++var) ((TArc*)li->At(var))->SetLineColor(col);
      } else ((TPolyLine*)fPolyList->At(ui))->SetLineColor(col);
   }
}


//______________________________________________________________________________
void TSpider::SetLineWidth(Width_t wid)
{
   //Set line width.

   UInt_t ui=0;

   TAttLine::SetLineWidth(wid);
   for(ui=0; ui<fNx*fNy;++ui){
      if(fSegmentDisplay){
         TList *li = (TList*)fPolyList->At(ui);
         for(UInt_t var=0;var<fNcols;++var) ((TArc*)li->At(var))->SetLineWidth(wid);
      } else ((TPolyLine*)fPolyList->At(ui))->SetLineWidth(wid);
   }
}


//______________________________________________________________________________
void TSpider::SetFillColor(Color_t col)
{
   // Set fill color.

   UInt_t ui=0;

   TAttFill::SetFillColor(col);
   for(ui=0; ui<fNx*fNy;++ui){
      if(fSegmentDisplay){
         TList *li = (TList*)fPolyList->At(ui);
         for(UInt_t var=0;var<fNcols;++var) ((TArc*)li->At(var))->SetFillColor(col);
      } else ((TPolyLine*)fPolyList->At(ui))->SetFillColor(col);
   }
}


//______________________________________________________________________________
void TSpider::SetFillStyle(Style_t sty)
{
   // Set fill style.

   UInt_t ui=0;

   TAttFill::SetFillStyle(sty);
   for(ui=0; ui<fNx*fNy;++ui){
      if(fSegmentDisplay){
         TList *li = (TList*)fPolyList->At(ui);
         for(UInt_t var=0;var<fNcols;++var) ((TArc*)li->At(var))->SetFillStyle(sty);
      } else ((TPolyLine*)fPolyList->At(ui))->SetFillStyle(sty);
   }
}


//______________________________________________________________________________
void TSpider::SetNdivRadial(Int_t ndiv)
{
   // Set number of radial divisions.

   if(fPolargram->GetNdivRadial() == ndiv) return;
   fPolargram->SetNdivRadial(ndiv);
}


//______________________________________________________________________________
void TSpider::SetNx(UInt_t nx)
{
   // Set the X number of subpads.

   if(fNx == nx || nx <= 0) return;
   fEntry = fCurrentEntries[0];

   UInt_t ui=0;
   Color_t lc;
   Style_t lt;
   Width_t lw;
   Color_t fc;
   Style_t fs;
   if(fAverageSlices){
      lc = fAverageSlices[0]->GetLineColor();
      lt = fAverageSlices[0]->GetLineStyle();
      lw = fAverageSlices[0]->GetLineWidth();
      fc = fAverageSlices[0]->GetFillColor();
      fs = fAverageSlices[0]->GetFillStyle();
   } else {
      lc = fAveragePoly->GetLineColor();
      lt = fAveragePoly->GetLineStyle();
      lw = fAveragePoly->GetLineWidth();
      fc = fAveragePoly->GetFillColor();
      fs = fAveragePoly->GetFillStyle();
   }

   if(fSegmentDisplay){
      for(ui=0; ui<fNx*fNy;++ui) ((TList*)fPolyList->At(ui))->Delete();
   }
   fPolyList->Delete();
   delete fPolyList;
   fPolyList = NULL;
   delete [] fCurrentEntries;
   fCurrentEntries = NULL;

   fNx = nx;

   fTree->SetScanField(fNx*fNy);
   SetCurrentEntries();
   if (fCanvas) {
      fCanvas->Clear();
      fCanvas->Divide(fNx,fNy);
   }

   for(ui=0; ui < fNx*fNy;++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      fPolargram->Draw("pn");
      fTree->LoadTree(fCurrentEntries[ui]);
      if(fSegmentDisplay){
         if(fDisplayAverage) DrawSlicesAverage("");
         DrawSlices("");
      } else {
         if(fDisplayAverage) DrawPolyAverage("");
         DrawPoly("");
      }
      AppendPad();
   }

   if(fAverageSlices){
      for(ui = 0;ui<fNcols;++ui){
         fAverageSlices[ui]->SetLineColor(lc);
         fAverageSlices[ui]->SetLineStyle(lt);
         fAverageSlices[ui]->SetLineWidth(lw);
         fAverageSlices[ui]->SetFillColor(fc);
         fAverageSlices[ui]->SetFillStyle(fs);
      }
   } else {
      fAveragePoly->SetLineColor(lc);
      fAveragePoly->SetLineStyle(lt);
      fAveragePoly->SetLineWidth(lw);
      fAveragePoly->SetFillColor(fc);
      fAveragePoly->SetFillStyle(fs);
   }
}


//______________________________________________________________________________
void TSpider::SetNy(UInt_t ny)
{
   // Set the Y number of subpads.

   if(fNy == ny || ny <= 0) return;
   fEntry = fCurrentEntries[0];

   UInt_t ui=0;
   Color_t lc;
   Style_t lt;
   Width_t lw;
   Color_t fc;
   Style_t fs;
   if(fAverageSlices){
      lc = fAverageSlices[0]->GetLineColor();
      lt = fAverageSlices[0]->GetLineStyle();
      lw = fAverageSlices[0]->GetLineWidth();
      fc = fAverageSlices[0]->GetFillColor();
      fs = fAverageSlices[0]->GetFillStyle();
   } else {
      lc = fAveragePoly->GetLineColor();
      lt = fAveragePoly->GetLineStyle();
      lw = fAveragePoly->GetLineWidth();
      fc = fAveragePoly->GetFillColor();
      fs = fAveragePoly->GetFillStyle();
   }

   if(fSegmentDisplay){
      for(ui=0; ui<fNx*fNy;++ui) ((TList*)fPolyList->At(ui))->Delete();
   }
   fPolyList->Delete();
   delete fPolyList;
   fPolyList = NULL;
   delete [] fCurrentEntries;
   fCurrentEntries = NULL;

   fNy = ny;

   fTree->SetScanField(fNx*fNy);
   SetCurrentEntries();
   if (fCanvas) {
      fCanvas->Clear();
      fCanvas->Divide(fNx,fNy);
   }

   for(ui=0; ui < fNx*fNy;++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      fPolargram->Draw("pn");
      fTree->LoadTree(fCurrentEntries[ui]);
      if(fSegmentDisplay){
         if(fDisplayAverage) DrawSlicesAverage("");
         DrawSlices("");
      } else {
         if(fDisplayAverage) DrawPolyAverage("");
         DrawPoly("");
      }
      AppendPad();
   }

   if(fAverageSlices){
      for(ui = 0;ui<fNcols;++ui){
         fAverageSlices[ui]->SetLineColor(lc);
         fAverageSlices[ui]->SetLineStyle(lt);
         fAverageSlices[ui]->SetLineWidth(lw);
         fAverageSlices[ui]->SetFillColor(fc);
         fAverageSlices[ui]->SetFillStyle(fs);
      }
   } else {
      fAveragePoly->SetLineColor(lc);
      fAveragePoly->SetLineStyle(lt);
      fAveragePoly->SetLineWidth(lw);
      fAveragePoly->SetFillColor(fc);
      fAveragePoly->SetFillStyle(fs);
   }
}


//______________________________________________________________________________
void TSpider::SetSegmentDisplay(Bool_t seg)
{
   // Set the segment display or not.

   if(seg == fSegmentDisplay) return;

   UInt_t ui=0;

   if(fSegmentDisplay){
      for(ui=0;ui<fNx*fNy;++ui){
         ((TList*)fPolyList->At(ui))->Delete();
      }
   }
   fPolyList->Delete();

   Color_t lc;
   Style_t lt;
   Width_t lw;
   Color_t fc;
   Style_t fs;
   if(fAverageSlices){
      lc = fAverageSlices[0]->GetLineColor();
      lt = fAverageSlices[0]->GetLineStyle();
      lw = fAverageSlices[0]->GetLineWidth();
      fc = fAverageSlices[0]->GetFillColor();
      fs = fAverageSlices[0]->GetFillStyle();
   } else {
      lc = fAveragePoly->GetLineColor();
      lt = fAveragePoly->GetLineStyle();
      lw = fAveragePoly->GetLineWidth();
      fc = fAveragePoly->GetFillColor();
      fs = fAveragePoly->GetFillStyle();
   }
   delete fPolyList;
   fPolyList = NULL;
   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui) delete fAverageSlices[ui];
   }
   delete [] fAverageSlices;
   fAverageSlices = NULL;
   delete fAveragePoly;
   fAveragePoly = NULL;

   for(ui=0;ui<fNx*fNy;++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      gPad->Clear();
   }

   fSegmentDisplay = seg;

   for(ui=0; ui < fNx*fNy;++ui){
      if (fCanvas) fCanvas->cd(ui+1);
      fPolargram->Draw("pn");
      fTree->LoadTree(fCurrentEntries[ui]);
      if(fSegmentDisplay){
         if(fDisplayAverage) DrawSlicesAverage("");
         DrawSlices("");
      } else {
         if(fDisplayAverage) DrawPolyAverage("");
         DrawPoly("");
      }
      AppendPad();
   }

   if(fAverageSlices){
      for(ui=0;ui<fNcols;++ui){
         fAverageSlices[ui]->SetLineColor(lc);
         fAverageSlices[ui]->SetLineStyle(lt);
         fAverageSlices[ui]->SetLineWidth(lw);
         fAverageSlices[ui]->SetFillColor(fc);
         fAverageSlices[ui]->SetFillStyle(fs);
      }
   } else {
      fAveragePoly->SetLineColor(lc);
      fAveragePoly->SetLineStyle(lt);
      fAveragePoly->SetLineWidth(lw);
      fAveragePoly->SetFillColor(fc);
      fAveragePoly->SetFillStyle(fs);
   }
   if (fCanvas) {
      fCanvas->Modified();
      fCanvas->Update();
   }
}




//______________________________________________________________________________
void TSpider::SetSelectionExpression(const char* selection)
{
   // Compile selection expression if there is one.

   if (selection && strlen(selection)) {
      fSelect = new TTreeFormula("Selection",selection,fTree);
   //         if (!fSelect) return -1;
   //         if (!fSelect->GetNdim()) { delete fSelect; return -1; }
      fFormulas->Add(fSelect);
   }
}


//______________________________________________________________________________
void TSpider::SetVariablesExpression(const char* varexp)
{
   // Compile the variables expression from the given string varexp.

   Int_t nch;
   fNcols=8;

   if (!varexp) return;
   TObjArray *leaves = fTree->GetListOfLeaves();
   UInt_t nleaves = leaves->GetEntriesFast();
   if (nleaves < fNcols) fNcols = nleaves;
   nch = strlen(varexp);

   // if varexp is empty, take first 8 columns by default
   Int_t allvar = 0;
   std::vector<TString> cnames;
   if (!strcmp(varexp, "*")) { fNcols = nleaves; allvar = 1; }
   if (nch == 0 || allvar) {
      UInt_t ncs = fNcols;
      fNcols = 0;
      for (UInt_t ui=0;ui<ncs;++ui) {
         TLeaf *lf = (TLeaf*)leaves->At(ui);
         if (lf->GetBranch()->GetListOfBranches()->GetEntries() > 0) continue;
         cnames.push_back(lf->GetName());
         fNcols++;
      }
      // otherwise select only the specified columns
   } else {
      fNcols = fSelector->SplitNames(varexp,cnames);
   }

   // Create the TreeFormula objects corresponding to each column
   for (UInt_t ui=0;ui<fNcols;ui++) {
      fFormulas->Add(new TTreeFormula("Var1",cnames[ui].Data(),fTree));
   }
}


//______________________________________________________________________________
void TSpider::SyncFormulas()
{
   // Create a TreeFormulaManager to coordinate the formulas.

   Int_t i;
   if (fFormulas->LastIndex()>=0) {
      if (fSelect) {
         if (fSelect->GetManager()->GetMultiplicity() > 0 ) {
            if(!fManager) fManager = new TTreeFormulaManager;
            for(i=0;i<=fFormulas->LastIndex();i++) {
               fManager->Add((TTreeFormula*)fFormulas->At(i));
            }
            fManager->Sync();
         }
      }
      for(i=0;i<=fFormulas->LastIndex();i++) {
         TTreeFormula *form = ((TTreeFormula*)fFormulas->At(i));
         switch( form->GetManager()->GetMultiplicity() ) {
            case  1:
            case  2:
            case -1:
               fForceDim = kTRUE;
               break;
            case  0:
               break;
         }

      }
   }
}


//______________________________________________________________________________
void TSpider::UpdateView()
{
   // Update the polylines or the arcs for the current entries.

   Double_t slice = 2*TMath::Pi()/fNcols;

   Double_t x,y,r;

   for(UInt_t pad=1;pad <= fNx*fNy;++pad){
      fTree->LoadTree(fCurrentEntries[pad-1]);
      for(UInt_t i=0;i<fNcols;++i){
         r = (((TTreeFormula*)fFormulas->At(i))->EvalInstance()-fMin[i])/(fMax[i]-fMin[i]);
         x=r*TMath::Cos(i*slice);
         y=r*TMath::Sin(i*slice);
         if(!fSegmentDisplay) ((TPolyLine*)fPolyList->At(pad-1))->SetPoint(i,x,y);
         else {
            ((TArc*)((TList*)fPolyList->At(pad-1))->At(i))->SetR1(r);
            ((TArc*)((TList*)fPolyList->At(pad-1))->At(i))->SetR2(r);
         }
      }
      x=(((TTreeFormula*)fFormulas->At(0))->EvalInstance()-fMin[0])/(fMax[0]-fMin[0]);
      y=0;
      if(!fSegmentDisplay) ((TPolyLine*)fPolyList->At(pad-1))->SetPoint(fNcols,x,y);
   }
}
