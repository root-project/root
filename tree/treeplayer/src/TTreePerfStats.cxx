// @(#)root/treeplayer:$Id$
// Author: Rene Brun 29/10/09

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreePerfStats                                                       //
//                                                                      //
// TTree I/O performance measurement                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TTreePerfStats.h"
#include "TROOT.h"
#include "Riostream.h"
#include "TFile.h"
#include "TTree.h"
#include "TAxis.h"
#include "TVirtualPad.h"
#include "TPaveText.h"
#include "TGraphErrors.h"
#include "TStopwatch.h"


ClassImp(TTreePerfStats)

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats() : TVirtualPerfStats()
{
   // default constructor (used when reading an object only)

   fName   = "";
   fTree   = 0;
   fNleaves=0;
   fFile   = 0;
   fGraph  = 0;
   fWatch  = 0;
   fPave   = 0;
   fTreeCacheSize = 0;
   fReadCalls     = 0;
   fReadaheadSize = 0;
   fBytesRead     = 0;
   fBytesReadExtra= 0;
   fRealTime      = 0;
   fCpuTime       = 0;
}

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats(const char *name, TTree *T) : TVirtualPerfStats()
{
   // Create a TTree I/O perf stats object.

   fName   = name;
   fTree   = T;
   fNleaves= T->GetListOfLeaves()->GetEntries();
   fFile   = T->GetCurrentFile();
   fGraph  = new TGraphErrors(0);
   fGraph->SetName("ioperf");
   fGraph->SetTitle(Form("%s/%s",fFile->GetName(),T->GetName()));
   fGraph->SetUniqueID(999999999);
   fWatch  = new TStopwatch();
   fWatch->Start();
   fPave  = 0;
   fTreeCacheSize = 0;
   fReadCalls     = 0;
   fReadaheadSize = 0;
   fBytesRead     = 0;
   fBytesReadExtra= 0;
   fRealTime      = 0;
   fCpuTime       = 0;
   gPerfStats = this;
}

//______________________________________________________________________________
TTreePerfStats::~TTreePerfStats()
{
   // Destructor
   
   fTree = 0;
   fFile = 0;
   delete fGraph;
   delete fPave;
   delete fWatch;
}

//______________________________________________________________________________
Int_t TTreePerfStats::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Return distance to one of the objects in the TTreePerfStats
   
   const Int_t kMaxDiff = 7;
   Int_t distance;
   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   if (px > puxmax) return 9999;
   if (py < puymax) return 9999;
   distance = fGraph->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {if (px > puxmin && py < puymin) gPad->SetSelected(fGraph); return distance;}
   distance = fPave->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {gPad->SetSelected(fPave);  return distance;}
   if (px > puxmax-300) return 2;
   return 999;
}

//______________________________________________________________________________
void TTreePerfStats::Draw(Option_t *option)
{
   // Draw the TTree I/O perf graph.
   // by default the graph is drawn with option "al"
   // Specify option ="ap" to show only the read blocks and not the line
   // connecting the blocks

   Finish();
   
   TString opt = option;
   if (strlen(option)==0) opt = "al";
   opt.ToLower();
   if (gPad) {
      if (!gPad->IsEditable()) gROOT->MakeDefCanvas();
      //the following statement is necessary in case one attempts to draw
      //a temporary histogram already in the current pad
      if (TestBit(kCanDelete)) gPad->GetListOfPrimitives()->Remove(this);
   } else {
      gROOT->MakeDefCanvas();
   }
   if (opt.Contains("a")) {
      gPad->Clear();
      gPad->SetGridx();
      gPad->SetGridy();
   }
   AppendPad(opt.Data());
}

//______________________________________________________________________________
void TTreePerfStats::ExecuteEvent(Int_t /*event*/, Int_t /*px*/, Int_t /*py*/)
{
   // Return distance to one of the objects in the TTreePerfStats
   
}

//______________________________________________________________________________
void TTreePerfStats::FileReadEvent(TFile *file, Int_t len, Double_t /*proctime*/)
{
   // Record TTree file read event.

   Long64_t offset = file->GetRelOffset();
   Int_t np = fGraph->GetN();
   Int_t entry = fTree->GetReadEntry();
   Double_t err = len/2.;
   fGraph->SetPoint(np,entry,offset-err);
   fGraph->SetPointError(np,0.01,len);
}

//______________________________________________________________________________
void TTreePerfStats::Finish()
{
   // When the run is finished this function must be called
   // to save the current parameters in the file and Tree in this object
   // the function is automatically called by Draw and Print
   
   if (fReadCalls)  return;  //has already been called
   if (!fFile)      return;
   if (!fTree)      return;
   fReadCalls     = fFile->GetReadCalls();
   fTreeCacheSize = fTree->GetCacheSize();
   fReadaheadSize = TFile::GetReadaheadSize();
   fBytesRead     = fFile->GetBytesRead();
   fBytesReadExtra= fFile->GetBytesReadExtra();
   fRealTime      = fWatch->RealTime();
   fCpuTime       = fWatch->CpuTime();
}
   

//______________________________________________________________________________
void TTreePerfStats::Paint(Option_t *option)
{
   // Draw the TTree I/O perf graph.

   fGraph->Paint(option);
   fGraph->GetXaxis()->SetTitle("Tree entry number");
   fGraph->GetYaxis()->SetTitle("file position");
   fGraph->GetYaxis()->SetTitleOffset(1.2);

   Double_t extra = 100.*fBytesReadExtra/fBytesRead;
   if (!fPave) {
      fPave = new TPaveText(.2,.55,.5,.88,"brNDC");
      fPave->SetTextAlign(12);
      fPave->AddText(Form("TreeCache = %d MBytes",fTreeCacheSize/1000000));
      fPave->AddText(Form("N leaves  = %d",fNleaves));
      fPave->AddText(Form("ReadTotal = %g MBytes",1e-6*fBytesRead));
      fPave->AddText(Form("ReadCalls = %d",fReadCalls));
      fPave->AddText(Form("ReadSize  = %g KBytes/read",0.001*fBytesRead/fReadCalls));
      fPave->AddText(Form("Readahead = %d KBytes",fReadaheadSize/1000));
      fPave->AddText(Form("Readextra = %5.2f per cent",extra));
      fPave->AddText(Form("Real Time = %7.3f seconds",fRealTime));
      fPave->AddText(Form("CPU  Time = %7.3f seconds",fCpuTime));
      fPave->AddText(Form("ReadRT    = %7.3f MBytes/s",1e-6*fBytesRead/fRealTime));
      fPave->AddText(Form("ReadCP    = %7.3f MBytes/s",1e-6*fBytesRead/fCpuTime));
   }
   fPave->Paint();
}

//______________________________________________________________________________
void TTreePerfStats::Print(Option_t * /*option*/) const
{
   // Print the TTree I/O perf stats.
   
   Double_t extra = 100.*fBytesReadExtra/fBytesRead;
   printf("TreeCache = %d MBytes\n",Int_t(fTreeCacheSize/1000000));
   printf("N leaves  = %d\n",fNleaves);
   printf("ReadTotal = %g MBytes\n",1e-6*fBytesRead);
   printf("ReadCalls = %d\n",fReadCalls);
   printf("ReadSize  = %g KBytes/read\n",0.001*fBytesRead/fReadCalls);
   printf("Readahead = %d KBytes\n",fReadaheadSize/1000);
   printf("Readextra = %5.2f per cent\n",extra);
   printf("Real Time = %7.3f seconds\n",fRealTime);
   printf("CPU  Time = %7.3f seconds\n",fCpuTime);
   printf("ReadRT    = %7.3f MBytes/s\n",1e-6*fBytesRead/fRealTime);
   printf("ReadCP    = %7.3f MBytes/s\n",1e-6*fBytesRead/fCpuTime);
}

//______________________________________________________________________________
void TTreePerfStats::SaveAs(const char *filename, Option_t * /*option*/) const
{
   // Save this object to filename
   
   TTreePerfStats *ps = (TTreePerfStats*)this;
   ps->Finish();
   ps->TObject::SaveAs(filename);
}

//______________________________________________________________________________
void TTreePerfStats::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TTreePerfStats::Class())) {
      out<<"   ";
   } else {
      out<<"   TTreePerfStats *";
   }
   out<<"ps = new TTreePerfStats();"<<endl;
   out<<"   ps->SetName("<<quote<<GetName()<<quote<<");"<<endl;
   out<<"   ps->SetTreeCacheSize("<<fTreeCacheSize<<");"<<endl;
   out<<"   ps->SetNleaves("<<fNleaves<<");"<<endl;
   out<<"   ps->SetReadCalls("<<fReadCalls<<");"<<endl;
   out<<"   ps->SetReadaheadSize("<<fReadaheadSize<<");"<<endl;
   out<<"   ps->SetBytesRead("<<fBytesRead<<");"<<endl;
   out<<"   ps->SetBytesReadExtra("<<fBytesReadExtra<<");"<<endl;
   out<<"   ps->SetRealTime("<<fRealTime<<");"<<endl;
   out<<"   ps->SetCpuTime("<<fCpuTime<<");"<<endl;

   Int_t npoints = fGraph->GetN();
   out<<"   TGraphErrors *psGraph = new TGraphErrors("<<npoints<<");"<<endl;
   out<<"   psGraph->SetName("<<quote<<fGraph->GetName()<<quote<<");"<<endl;
   out<<"   psGraph->SetTitle("<<quote<<fGraph->GetTitle()<<quote<<");"<<endl;
   out<<"   ps->SetGraph(psGraph);"<<endl;
   for (Int_t i=0;i<npoints;i++) {
      out<<"   psGraph->SetPoint("<<i<<","<<fGraph->GetX()[i]<<","<<fGraph->GetY()[i]<<");"<<endl;
      out<<"   psGraph->SetPointError("<<i<<",0,"<<fGraph->GetEY()[i]<<");"<<endl;
   }

   out<<"   ps->Draw("<<quote<<option<<quote<<");"<<endl;
}
