// @(#)root/sessionviewer:$Id$
// Author: Anna Kreshuk 18/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressMemoryPlot                                             //
//                                                                      //
// This class implements a dialog, used to display the memory footprint //
// on the proof workers and master. For the workers, memory is plotted  //
// as a function of number of events processed. For the master, it is   //
// plotted as a function of number of objects merged                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofProgressMemoryPlot.h"
#include "TProofProgressDialog.h"
#include "TRootEmbeddedCanvas.h"
#include "TCanvas.h"
#include "TGListBox.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "TProofLog.h"
#include "TUrl.h"
#include "TProof.h"
#include "TError.h"
#include "TGFrame.h"
#include "TMacro.h"
#include "TObjString.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TAxis.h"

#define kMemValuePos 8
#define kMemValuePosMaster 8
#define kEventNumberPos 13

ClassImp(TProofProgressMemoryPlot)

//_________________________________________________________________________
TProofProgressMemoryPlot::TProofProgressMemoryPlot(TProofProgressDialog *d,
                                                   Int_t w, Int_t h)
                         : TGTransientFrame(gClient->GetRoot(),
                                            gClient->GetRoot(), w, h)
{
   // Main constructor

   fDialog = d;
   fProofLog = 0;
   fWPlot = 0;
   fMPlot = 0;
   fAPlot = 0;
   fFullLogs = kFALSE;

   // use hierarchical cleaning
   SetCleanup(kDeepCleanup);
   TGHorizontalFrame *htotal = new TGHorizontalFrame(this, w, h);
   //The frame for choosing workers
   TGVerticalFrame *vworkers = new TGVerticalFrame(htotal);
   TGLabel *label1 = new TGLabel(vworkers,"Choose workers:");

   //The list of workers
   fWorkers = BuildLogList(vworkers);
   fWorkers->Resize(102,52);
   fWorkers->SetMultipleSelections(kTRUE);

   //The SelectAll/ClearAll button
   TGPopupMenu *pm = new TGPopupMenu(gClient->GetRoot());
   pm->AddEntry("Select All", 0);
   pm->AddEntry("Clear All", 1);

   fAllWorkers = new TGSplitButton(vworkers, new TGHotString("Select ...            "), pm);
   fAllWorkers->Connect("ItemClicked(Int_t)", "TProofProgressMemoryPlot", this,
                        "Select(Int_t)");
   fAllWorkers->SetSplit(kFALSE);
   //select all for the first display
   Select(1);
   fWorkers->Select(0, kTRUE);
   fWorkers->Select(1, kTRUE);

   fPlot = new TGTextButton(vworkers, "Plot");
   fPlot->Connect("Clicked()", "TProofProgressMemoryPlot", this, "DoPlot()");
   vworkers->AddFrame(label1, new TGLayoutHints(kLHintsLeft | kLHintsTop, 7, 2, 5, 2));
   vworkers->AddFrame(fAllWorkers, new TGLayoutHints(kLHintsExpandX | kLHintsTop, 5, 2, 2, 2));
   vworkers->AddFrame(fWorkers, new TGLayoutHints(kLHintsExpandX | kLHintsTop | kLHintsExpandY, 2, 2, 5, 2));
   vworkers->AddFrame(fPlot, new TGLayoutHints(kLHintsExpandX | kLHintsBottom, 2, 2, 2, 2));
   htotal->AddFrame(vworkers, new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandY, 2, 2, 2, 2));

   fWorkersPlot = new TRootEmbeddedCanvas("WorkersCanvas", htotal, 300, 300);
   htotal->AddFrame(fWorkersPlot, new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 2, 2, 2, 2));
   fMasterPlot = new TRootEmbeddedCanvas("MasterCanvas", htotal, 300, 300);
   htotal->AddFrame(fMasterPlot, new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 2, 2, 2 ,2));

   AddFrame(htotal, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 2, 2, 2, 2));
   char title[256] = {0};
   strcpy(title,Form("PROOF Memory Consumption: %s",
                     (fDialog->fProof ? fDialog->fProof->GetMaster() : "<dummy>")));
   SetWindowName(title);
   SetIconName(title);

   MapSubwindows();

   Resize();

   Window_t wdummy;
   int ax, ay;
   gVirtualX->TranslateCoordinates(GetParent()->GetId(), fDialog->fDialog->GetId(),
       (Int_t)(((TGFrame *)GetParent())->GetWidth() + w),
       (Int_t)(((TGFrame *)GetParent())->GetHeight()- 3*h/2), ax, ay, wdummy);
   Move(ax, ay);

   //Popup();
   MapWindow();
}

//______________________________________________________________________________
TProofProgressMemoryPlot::~TProofProgressMemoryPlot()
{
   // Destructor

   if (fProofLog){
      delete fProofLog;
      fProofLog = 0;
   }
   if (fMPlot){
      delete fMPlot;
      fMPlot = 0;
   }
   if (fWPlot){
      delete fWPlot;
      fWPlot = 0;
   }

   fProofLog = 0;
   fDialog->fMemWindow = 0;

}

//______________________________________________________________________________
TGListBox* TProofProgressMemoryPlot::BuildLogList(TGFrame *parent)
{
   // Build the list of workers. For this, extract the logs and take the names
   // of TProofLogElements

   TGListBox *c = new TGListBox(parent);
   c->AddEntry("average", 0);

   SafeDelete(fProofLog);

   fProofLog = TProof::Mgr(fDialog->fSessionUrl.Data())->GetSessionLogs(0, 0, "Svc.*Memory");
   if (fDialog->fStatus==TProofProgressDialog::kRunning) {
      fFullLogs = kFALSE;
   } else {
      fFullLogs = kTRUE;
   }

   TList *elem = fProofLog->GetListOfLogs();
   TIter next(elem);
   TProofLogElem *pe = 0;

   TString buf;
   Int_t is = 1;
   while ((pe=(TProofLogElem*)next())){
      TUrl url(pe->GetTitle());
      buf = Form("%s %s", pe->GetName(), url.GetHost());
      c->AddEntry(buf.Data(), is);
      is++;
   }
   return c;

}

//______________________________________________________________________________
void TProofProgressMemoryPlot::Clear(Option_t *)
{
   // Clear the canvases

   if (fWorkersPlot)
      fWorkersPlot->GetCanvas()->Clear();
   if (fMasterPlot)
      fMasterPlot->GetCanvas()->Clear();
}

//______________________________________________________________________________
void TProofProgressMemoryPlot::DoPlot()
{
   // Draw the plot from the logs

   Clear();

   if (!fProofLog || !fFullLogs ||
      (fDialog && fDialog->fStatus == TProofProgressDialog::kRunning)){

      SafeDelete(fProofLog);
      if (fDialog) {
         fProofLog = TProof::Mgr(fDialog->fSessionUrl.Data())->GetSessionLogs(0, 0, "Svc.*Memory");
         if (fDialog->fStatus==TProofProgressDialog::kRunning) {
            fFullLogs = kFALSE;
         } else {
            fFullLogs = kTRUE;
         }
      } else {
         Error("DoPlot", "TProofProgessDialog instance undefined - protocol error?");
         return;
      }
   }

   // Make sure we have something to parse
   if (!fProofLog) {
      Error("DoPlot", "could not get a valid instance of TProofLog");
      return;
   }

   char name[512]; //should be long enough

   TList *elem = fProofLog->GetListOfLogs();
   if (!elem) {Error("DoPlot", "No log elements\n"); return;}
   TIter next(elem);
   TProofLogElem *ple=0;

   Int_t iwelem = 0;
   Int_t imelem = 0;
   TGraph *gr=0;

   TList *selected = new TList;
   fWorkers->GetSelectedEntries(selected);
   TIter nextworker(selected);
   TGTextLBEntry *selworker;
   TLegend *legw = 0;
   TLegend *legm = 0;

   //delete the current multigraphs
   if (fWPlot){
      delete fWPlot;
      fWPlot = 0;
   }
   if (fMPlot) {
      delete fMPlot;
      fMPlot = 0;
   }

   //loop over the selected workers in the list
   Int_t max = -1;
   Int_t min = -1;
   while ((selworker=(TGTextLBEntry*)nextworker())){

      sprintf(name, "%s", selworker->GetText()->GetString());
      char *token;
      token = strtok(name, " ");
      if (!strcmp(token, "average")) { //change that to id comparison later
         gr = DoAveragePlot(max, min);
         if (gr && gr->GetN()>0){
            if (!fWPlot){
               fWPlot = new TMultiGraph();
               if (!legw){
                  legw = new TLegend(0.1, 0.7, 0.4, 0.9);
                  legw->SetHeader("Workers");
               }
            }
            gr->SetMarkerColor(1);
            gr->SetMarkerStyle(2);
            gr->SetMarkerSize(1);
            gr->SetLineWidth(2);
            gr->SetLineColor(1);
            fWPlot->Add(gr, "l");
            legw->AddEntry(gr, token, "l");
         }
         TProofLogElem *pltemp = (TProofLogElem*)elem->At(min+1);
         gr = DoWorkerPlot(pltemp);
         if (gr && gr->GetN()>0){
            gr->SetLineWidth(2);
            gr->SetLineColor(2);
            gr->SetLineStyle(3);
            fWPlot->Add(gr, "l");
            legw->AddEntry(gr, Form("%s - min", pltemp->GetName()) , "l");
         }
         pltemp = (TProofLogElem*)elem->At(max+1);
         gr = DoWorkerPlot(pltemp);
         if (gr && gr->GetN()>0){
            gr->SetLineWidth(2);
            gr->SetLineColor(2);
            gr->SetLineStyle(2);
            fWPlot->Add(gr, "l");
            legw->AddEntry(gr, Form("%s - max", pltemp->GetName()), "l");
         }


         continue;
      }


      ple = (TProofLogElem*)elem->FindObject(token);
      const char *role = ple->GetRole();
      if (role[0]=='w'){
         //role should be equal to "worker", only check the 1st char

         gr = DoWorkerPlot(ple);
         if (gr && gr->GetN()>0){
            if (!fWPlot){
               fWPlot = new TMultiGraph();
               if (!legw){
                  legw = new TLegend(0.1, 0.7, 0.4, 0.9);
                  legw->SetHeader("Workers");
               }
            }
            gr->SetLineWidth(2);
            gr->SetLineColor(iwelem+3);
            fWPlot->Add(gr, "l");
            legw->AddEntry(gr, token, "l");
            iwelem++;
         }
      } else {
         //a master or submaster log
         //display without meaningful labels for now
         gr = DoMasterPlot(ple);
         if (gr && gr->GetN()>0){
            if (!fMPlot){
               fMPlot = new TMultiGraph();
               legm = new TLegend(0.1, 0.7, 0.4, 0.9);
               legm->SetHeader("Master");
            }
            gr->SetLineWidth(2);
            gr->SetLineColor(imelem+1);
            fMPlot->Add(gr, "l");
            legm->AddEntry(gr, token, "l");
            imelem++;
         }
      }

   }

   if (fWPlot){
      fWorkersPlot->GetCanvas()->cd();
      fWPlot->Draw("a");
      fWPlot->GetXaxis()->SetTitle("Events Processed");
      fWPlot->GetYaxis()->SetTitle("MBytes");
      if (legw) legw->Draw();

   }
   if (fMPlot) {
      fMasterPlot->GetCanvas()->cd();
      fMPlot->Draw("a");
      fMPlot->GetXaxis()->SetTitle("Objects Merged");
      fMPlot->GetYaxis()->SetTitle("MBytes");
      if (legm) legm->Draw();
   }
   fWorkersPlot->GetCanvas()->Update();
   fMasterPlot->GetCanvas()->Update();

   delete selected;
}

//______________________________________________________________________________
TGraph *TProofProgressMemoryPlot::DoAveragePlot(Int_t &max_el, Int_t &min_el)
{
   // Create the average plots

   TList *elem = fProofLog->GetListOfLogs();
   if (!elem) {
      Error("DoAveragePlot", "Empty log");
      return 0;
   }
   TIter next(elem);

   TProofLogElem *ple=0;
   Double_t max_av = 0;
   Double_t min_av = 10E9;

   Long64_t maxevent = 0;
   Long64_t step = -1;
   TObjString *curline = 0;
   TObjString *prevline = 0;
   TObjString *curevent = 0;
   Long64_t curevent_value;
   Long64_t prevevent_value;
   Long64_t *last = new Long64_t[elem->GetEntries()];
   TObjArray *parts = 0;
   TString token;
   Int_t ielem=0;
   while ((ple = (TProofLogElem*)next())){
      //find the maximal entry processed in the last query
      const char *role = ple->GetRole();
      if (role[0]!='w') continue; //skip the master log
      TList *lines = ple->GetMacro()->GetListOfLines();
      if (!lines || lines->GetSize() <= 0) continue;
      curline = (TObjString*)lines->Last();
      parts = curline->String().Tokenize(" ");
      curevent = (TObjString*)parts->At(kEventNumberPos);
      curevent_value = curevent->String().Atoll();
      if (maxevent < curevent_value) maxevent = curevent_value;
      last[ielem]=curevent_value;
      parts->Delete();
      delete parts;
      parts = 0;
      if (step < 0) {
         //find the step
         prevline = (TObjString*)lines->Before(curline);
         parts = prevline->String().Tokenize(" ");
         curevent = (TObjString*)parts->At(kEventNumberPos);
         prevevent_value = curevent->String().Atoll();
         step = curevent_value - prevevent_value;
         parts->Delete();
         delete parts;
         parts = 0;
      }
      ielem++;
   }
   Int_t maxlines = Int_t(maxevent/(1.*step));
   //transform the array of last event numbers to an array of numbers of lines
   for (Int_t i=0; i<ielem; i++){
      last[i] /= step;
   }

   Double_t *av_mem = new Double_t[maxlines];
   Int_t *nw = new Int_t[maxlines];
   for (Int_t i=0; i<maxlines; i++){
      av_mem[i]=0;
      nw[i]=0;
   }
   next.Reset();
   ielem=0;
   Int_t iline=0;
   Double_t tempval;
   Double_t cur_av;
   Long64_t nev;
   Long64_t nextevent = Long64_t(10E16);
   while ((ple = (TProofLogElem*)next())){
      const char *role = ple->GetRole();
      if (role[0]!='w') continue;
      TList *lines = ple->GetMacro()->GetListOfLines();
      if (!lines || lines->GetSize() <= 0) continue;
      TIter prev(lines, kIterBackward);
      nev = 0;
      nextevent = Long64_t(10E16);
      iline=0;
      cur_av = 0;
      while ((curline = (TObjString*)prev()) && iline<last[ielem]){
         // a backward loop, so that only the last query is counted
         Int_t from = 0;
         Int_t iword = 0;
         while (curline->String().Tokenize(token, from, " ")){
            if (iword==kMemValuePos){
               tempval = token.Atof();
               av_mem[last[ielem] -1 - iline] += tempval; //last[ielem] is the number of lines for
               nw[last[ielem] -1 - iline]++;              //this query and this element
               cur_av += tempval/last[ielem];
               // printf("added value %f at position %lld\n", tempval, last[ielem]-1-iline);
            }
            iword++;
         }
         iline++;
      }
      if (cur_av > max_av){
         max_av = cur_av;
         max_el = ielem;
      }
      if (cur_av < min_av){
         min_av = cur_av;
         min_el = ielem;
      }
      ielem++;
   }

   TGraph *gr = new TGraph(maxlines);
   //make an average
   for (Int_t i=0; i<maxlines; i++){
      gr->SetPoint(i, (i+1)*step, av_mem[i]/(nw[i]*1024.));
   }
   delete [] av_mem;
   av_mem = 0;
   delete [] nw;
   nw = 0;
   return gr;

}

//______________________________________________________________________________
TGraph *TProofProgressMemoryPlot::DoWorkerPlot(TProofLogElem *ple)
{
   // Make a memory consumption graph for a worker

   TObjString *curline;
   TList *lines = ple->GetMacro()->GetListOfLines();
   if (!lines) {
      //the log is empty so far
      return 0;
   }

   //find the last event value
   curline = (TObjString*)lines->Last();
   TObjArray *parts = curline->String().Tokenize(" ");
   TObjString *lastevent = (TObjString*)parts->At(kEventNumberPos);
   Long64_t lastevent_value = lastevent->String().Atoll();
   parts->Delete();
   delete parts;
   parts = 0;

   //find the step
   TObjString *prevline = (TObjString*)lines->Before(curline);
   parts = prevline->String().Tokenize(" ");
   lastevent = (TObjString*)parts->At(kEventNumberPos);
   Long64_t prevevent_value = lastevent->String().Atoll();
   Long64_t step = lastevent_value - prevevent_value;
   parts->Delete();
   delete parts;
   parts = 0;

   Int_t nlines = lastevent_value/step;
   TGraph *gr = new TGraph(nlines);

   TIter prevl(lines, kIterBackward);
   Int_t iline = 0;
   TString token;
   Double_t tempval;
   while ((curline = (TObjString*)prevl()) && iline<nlines){
      //iterate backwards so that only lines for the last query are taken
      Int_t from = 0;
      Int_t iword = 0;
      while (curline->String().Tokenize(token, from, " ")){
         if (iword==kMemValuePos){
            tempval = token.Atof();
            gr->SetPoint(nlines-1-iline,lastevent_value-iline*step, tempval/1024.);
            // printf("setting point %d x=%f, y=%lld\n", nlines-1-iline, tempval/1024., lastevent_value-iline*step);
         }
         iword++;
      }
      iline++;
   }

   return gr;
}

//______________________________________________________________________________
TGraph *TProofProgressMemoryPlot::DoMasterPlot(TProofLogElem *ple)
{
   // a master or submaster log
   // display without meaningful labels for now

   TList *lines = ple->GetMacro()->GetListOfLines();
   TIter prevline(lines, kIterBackward);
   Int_t iline=0;
   TObjString *curline;
   //count the number of lines that belong to the last query
   while ((curline = (TObjString*)prevline())) {
      if (curline->String().Contains("Start")) break;
      iline++;
   }

   Int_t nlines = iline;
   TString token;
   Double_t tempval;
   TGraph *gr = new TGraph(nlines);
   prevline.Reset();
   iline = 0;
   while ((curline = (TObjString*)prevline()) && iline<nlines) {
   //iterate backwards so that only lines for the last query are taken
   Int_t from = 0;
   Int_t iword = 0;
   while (curline->String().Tokenize(token, from, " ")){
      if (iword==kMemValuePosMaster){
         tempval = token.Atof();
         gr->SetPoint(nlines-iline, nlines-iline, tempval/1024.);
         //printf("setting point %d %d %f\n", nlines-iline, nlines-iline, tempval/1024.);
         break;
      }
      iword++;
   }
   iline++;
   }
   return gr;
}

//______________________________________________________________________________
void TProofProgressMemoryPlot::Select(Int_t id)
{
   //actions of select all/clear all button

   Int_t nen = fWorkers->GetNumberOfEntries();
   Bool_t sel = id ? 0 : 1;

   for (Int_t ie=0; ie<nen; ie++) {
      fWorkers->Select(ie, sel);
   }
}
