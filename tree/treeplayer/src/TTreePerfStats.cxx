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
//                                                                      
//                       TTreePerfStats                                                       
//                                                                      
//        TTree I/O performance measurement. see example of use below.
//
// The function FileReadEvent is called from TFile::ReadBuffer.
// For each call the following information is stored in fGraphIO
//     - x[i]  = Tree entry number
//     - y[i]  = file position
//     - ey[i] = 0.001*number of bytes read
// For each call the following information is stored in fGraphTime
//     - x[i]  = Tree entry number
//     - y[i]  = Time now
//     - ey[i] = 0.5*readtime, eg timenow - start
// The TTreePerfStats object can be saved in a ROOT file in such a way that
// its inspection can be done outside the job that generated it.
//
//       Example of use                                                 
// {
//   TFile *f = TFile::Open("RelValMinBias-GEN-SIM-RECO.root");
//   T = (TTree*)f->Get("Events");
//   Long64_t nentries = T->GetEntries();
//   T->SetCacheSize(10000000);
//   T->SetCacheEntryRange(0,nentries);
//   T->AddBranchToCache("*");
//
//   TTreePerfStats *ps= new TTreePerfStats("ioperf",T);
//
//   for (Int_t i=0;i<nentries;i++) {
//      T->GetEntry(i);
//   }
//   ps->SaveAs("cmsperf.root");
// }
//
// then, in a root interactive session, one can do:
//    root > TFile f("cmsperf.root");
//    root > ioperf->Draw();
//    root > ioperf->Print();
//
// The Draw or Print functions print the following information:
//   TreeCache = TTree cache size in MBytes
//   N leaves  = Number of leaves in the TTree
//   ReadTotal = Total number of zipped bytes read
//   ReadUnZip = Total number of unzipped bytes read
//   ReadCalls = Total number of disk reads
//   ReadSize  = Average read size in KBytes
//   Readahead = Readahead size in KBytes
//   Readextra = Readahead overhead in percent
//   Real Time = Real Time in seconds
//   CPU  Time = CPU Time in seconds
//   Disk Time = Real Time spent in pure raw disk IO
//   Disk IO   = Raw disk IO speed in MBytes/second
//   ReadUZRT  = Unzipped MBytes per RT second
//   ReadUZCP  = Unipped MBytes per CP second
//   ReadRT    = Zipped MBytes per RT second
//   ReadCP    = Zipped MBytes per CP second
//
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
#include "TGaxis.h"
#include "TTimeStamp.h"

const Double_t kScaleTime = 1e-20;

ClassImp(TTreePerfStats)

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats() : TVirtualPerfStats()
{
   // default constructor (used when reading an object only)

   fName      = "";
   fTree      = 0;
   fNleaves   = 0;
   fFile      = 0;
   fGraphIO   = 0;
   fGraphTime = 0;
   fWatch     = 0;
   fPave      = 0;
   fTreeCacheSize = 0;
   fReadCalls     = 0;
   fReadaheadSize = 0;
   fBytesRead     = 0;
   fBytesReadExtra= 0;
   fRealNorm      = 0;
   fRealTime      = 0;
   fCpuTime       = 0;
   fDiskTime      = 0;
   fCompress      = 0;
   fTimeAxis      = 0;
}

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats(const char *name, TTree *T) : TVirtualPerfStats()
{
   // Create a TTree I/O perf stats object.

   fName   = name;
   fTree   = T;
   fNleaves= T->GetListOfLeaves()->GetEntries();
   fFile   = T->GetCurrentFile();
   fGraphIO  = new TGraphErrors(0);
   fGraphIO->SetName("ioperf");
   fGraphIO->SetTitle(Form("%s/%s",fFile->GetName(),T->GetName()));
   fGraphIO->SetUniqueID(999999999);
   fGraphTime = new TGraphErrors(0);   
   fGraphTime->SetLineColor(kRed);
   fGraphTime->SetName("iotime");
   fGraphTime->SetTitle("Real time vs entries");
   fWatch  = new TStopwatch();
   fWatch->Start();
   fPave  = 0;
   fTreeCacheSize = 0;
   fReadCalls     = 0;
   fReadaheadSize = 0;
   fBytesRead     = 0;
   fBytesReadExtra= 0;
   fRealNorm      = 0;
   fRealTime      = 0;
   fCpuTime       = 0;
   fDiskTime      = 0;
   fTimeAxis      = 0;
   fCompress      = (T->GetTotBytes()+0.00001)/T->GetZipBytes();
   gPerfStats = this;
}

//______________________________________________________________________________
TTreePerfStats::~TTreePerfStats()
{
   // Destructor
   
   fTree = 0;
   fFile = 0;
   delete fGraphIO;
   delete fGraphTime;
   delete fPave;
   delete fWatch;
}

//______________________________________________________________________________
Int_t TTreePerfStats::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Return distance to one of the objects in the TTreePerfStats
   
   const Int_t kMaxDiff = 7;
   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   if (py < puymax) return 9999;
   //on the fGraphIO ?
   Int_t distance = fGraphIO->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {if (px > puxmin && py < puymin) gPad->SetSelected(fGraphIO); return distance;}
   // on the fGraphTime ?
   distance = fGraphTime->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {if (px > puxmin && py < puymin) gPad->SetSelected(fGraphTime); return distance;}
   // on the pave ?
   distance = fPave->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {gPad->SetSelected(fPave);  return distance;}
   // on the time axis ?
   distance = fTimeAxis->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {gPad->SetSelected(fTimeAxis);  return distance;}
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
      gPad->SetLeftMargin(0.35);
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
void TTreePerfStats::FileReadEvent(TFile *file, Int_t len, Double_t start)
{
   // Record TTree file read event.
   // start is the TimeStamp before reading
   // len is the number of bytes read

   Long64_t offset = file->GetRelOffset();
   Int_t np = fGraphIO->GetN();
   Int_t entry = fTree->GetReadEntry();
   fGraphIO->SetPoint(np,entry,offset);
   fGraphIO->SetPointError(np,0.001,0.001*len);
   Double_t tnow = TTimeStamp();
   Double_t dtime = tnow-start;
   fDiskTime += dtime;
   fGraphTime->SetPoint(np,entry,tnow);
   fGraphTime->SetPointError(np,0.001,0.5*dtime);
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
   Int_t npoints  = fGraphIO->GetN();
   if (!npoints) return;
   Double_t ymax  = fGraphIO->GetY()[npoints-1];
   Double_t tmax  = fGraphTime->GetY()[npoints-1];
   Double_t t0    = fGraphTime->GetY()[0];
   if (tmax <= t0) tmax = t0+1;
   fRealNorm      = ymax/(tmax-t0);
   // we normalize the fGraphTime such that it can be drawn on top of fGraphIO
   for (Int_t i=0;i<npoints;i++) {
      fGraphTime->GetY()[i]  -= t0;
      fGraphTime->GetY()[i]  *= fRealNorm;
      fGraphTime->GetEY()[i] *= fRealNorm;
   }
}
   

//______________________________________________________________________________
void TTreePerfStats::Paint(Option_t *option)
{
   // Draw the TTree I/O perf graph.

   Int_t npoints  = fGraphIO->GetN();
   if (!npoints) return;
   fGraphIO->GetXaxis()->SetTitle("Tree entry number");
   fGraphIO->GetYaxis()->SetTitle("file position");
   fGraphIO->GetYaxis()->SetTitleOffset(1.2);
   fGraphIO->GetXaxis()->SetLabelSize(0.03);
   fGraphIO->GetYaxis()->SetLabelSize(0.03);
   fGraphIO->Paint(option);
   
   //superimpose the time info (max 10 points)
   if (fGraphTime) {
      fGraphTime->Paint("l");
      if (!fTimeAxis) {
         Double_t uxmax = gPad->GetUxmax();
         Double_t uymax = gPad->GetUymax();
         Double_t tmax  = fGraphTime->GetY()[npoints-1]/fRealNorm;
         fTimeAxis = new TGaxis(uxmax,0,uxmax,uymax,0.,tmax,510,"+L");
         fTimeAxis->SetName("axisTime");
         fTimeAxis->SetLineColor(kRed);
         fTimeAxis->SetTitle("RealTime (s)");
         fTimeAxis->SetTitleColor(kRed);
         fTimeAxis->SetTitleOffset(1.2);
         fTimeAxis->SetLabelSize(0.03);
         fTimeAxis->SetLabelColor(kRed);
      }
      fTimeAxis->Paint();
   }

   Double_t extra = 100.*fBytesReadExtra/fBytesRead;
   if (!fPave) {
      fPave = new TPaveText(.01,.10,.24,.90,"brNDC");
      fPave->SetTextAlign(12);
      fPave->AddText(Form("TreeCache = %d MBytes",fTreeCacheSize/1000000));
      fPave->AddText(Form("N leaves  = %d",fNleaves));
      fPave->AddText(Form("ReadTotal = %g MBytes",1e-6*fBytesRead));
      fPave->AddText(Form("ReadUnZip = %g MBytes",1e-6*fBytesRead*fCompress));
      fPave->AddText(Form("ReadCalls = %d",fReadCalls));
      fPave->AddText(Form("ReadSize  = %7.3f KBytes/read",0.001*fBytesRead/fReadCalls));
      fPave->AddText(Form("Readahead = %d KBytes",fReadaheadSize/1000));
      fPave->AddText(Form("Readextra = %5.2f per cent",extra));
      fPave->AddText(Form("Real Time = %7.3f seconds",fRealTime));
      fPave->AddText(Form("CPU  Time = %7.3f seconds",fCpuTime));
      fPave->AddText(Form("Disk Time = %7.3f seconds",fDiskTime));
      fPave->AddText(Form("Disk IO   = %7.3f MBytes/s",1e-6*fBytesRead/fDiskTime));
      fPave->AddText(Form("ReadUZRT  = %7.3f MBytes/s",1e-6*fCompress*fBytesRead/fRealTime));
      fPave->AddText(Form("ReadUZCP  = %7.3f MBytes/s",1e-6*fCompress*fBytesRead/fCpuTime));
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
   printf("ReadUnZip = %g MBytes\n",1e-6*fBytesRead*fCompress);
   printf("ReadCalls = %d\n",fReadCalls);
   printf("ReadSize  = %7.3f KBytes/read\n",0.001*fBytesRead/fReadCalls);
   printf("Readahead = %d KBytes\n",fReadaheadSize/1000);
   printf("Readextra = %5.2f per cent\n",extra);
   printf("Real Time = %7.3f seconds\n",fRealTime);
   printf("CPU  Time = %7.3f seconds\n",fCpuTime);
   printf("Disk Time = %7.3f seconds\n",fDiskTime);
   printf("Disk IO   = %7.3f MBytes/s\n",1e-6*fBytesRead/fDiskTime);
   printf("ReadUZRT  = %7.3f MBytes/s\n",1e-6*fCompress*fBytesRead/fRealTime);
   printf("ReadUZCP  = %7.3f MBytes/s\n",1e-6*fCompress*fBytesRead/fCpuTime);
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
   out<<"   ps->SetRealNorm("<<fRealNorm<<");"<<endl;
   out<<"   ps->SetRealTime("<<fRealTime<<");"<<endl;
   out<<"   ps->SetCpuTime("<<fCpuTime<<");"<<endl;
   out<<"   ps->SetDiskTime("<<fDiskTime<<");"<<endl;
   out<<"   ps->SetCompress("<<fCompress<<");"<<endl;

   Int_t i, npoints = fGraphIO->GetN();
   out<<"   TGraphErrors *psGraphIO = new TGraphErrors("<<npoints<<");"<<endl;
   out<<"   psGraphIO->SetName("<<quote<<fGraphIO->GetName()<<quote<<");"<<endl;
   out<<"   psGraphIO->SetTitle("<<quote<<fGraphIO->GetTitle()<<quote<<");"<<endl;
   out<<"   ps->SetGraphIO(psGraphIO);"<<endl;
   fGraphIO->SaveFillAttributes(out,"psGraphIO",0,1001);
   fGraphIO->SaveLineAttributes(out,"psGraphIO",1,1,1);
   fGraphIO->SaveMarkerAttributes(out,"psGraphIO",1,1,1);
   for (i=0;i<npoints;i++) {
      out<<"   psGraphIO->SetPoint("<<i<<","<<fGraphIO->GetX()[i]<<","<<fGraphIO->GetY()[i]<<");"<<endl;
      out<<"   psGraphIO->SetPointError("<<i<<",0,"<<fGraphIO->GetEY()[i]<<");"<<endl;
   }
   npoints = fGraphTime->GetN();
   out<<"   TGraphErrors *psGraphTime = new TGraphErrors("<<npoints<<");"<<endl;
   out<<"   psGraphTime->SetName("<<quote<<fGraphTime->GetName()<<quote<<");"<<endl;
   out<<"   psGraphTime->SetTitle("<<quote<<fGraphTime->GetTitle()<<quote<<");"<<endl;
   out<<"   ps->SetGraphTime(psGraphTime);"<<endl;
   fGraphTime->SaveFillAttributes(out,"psGraphTime",0,1001);
   fGraphTime->SaveLineAttributes(out,"psGraphTime",1,1,1);
   fGraphTime->SaveMarkerAttributes(out,"psGraphTime",1,1,1);
   for (i=0;i<npoints;i++) {
      out<<"   psGraphTime->SetPoint("<<i<<","<<fGraphTime->GetX()[i]<<","<<fGraphTime->GetY()[i]<<");"<<endl;
      out<<"   psGraphTime->SetPointError("<<i<<",0,"<<fGraphTime->GetEY()[i]<<");"<<endl;
   }

   out<<"   ps->Draw("<<quote<<option<<quote<<");"<<endl;
}
