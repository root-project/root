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
//     - y[i]  = 1e-6*(file position)
//     - ey[i] = 1e-9*number of bytes read
// For each call the following information is stored in fGraphTime
//     - x[i]  = Tree entry number
//     - y[i]  = Time now
//     - ey[i] = readtime, eg timenow - start
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
//   NOTE1 : The ReadTotal value indicates the effective number of zipped bytes
//           returned to the application. The physical number of bytes read
//           from the device (as measured for example with strace) is
//           ReadTotal +ReadTotal*Readextra/100. Same for ReadSize.
//
//   NOTE2 : A consequence of NOTE1, the Disk I/O speed corresponds to the effective
//           number of bytes returned to the application per second.
//           The Physical disk speed is DiskIO + DiskIO*ReadExtra/100.
//
//////////////////////////////////////////////////////////////////////////


#include "TTreePerfStats.h"
#include "TROOT.h"
#include "TSystem.h"
#include "Riostream.h"
#include "TFile.h"
#include "TTree.h"
#include "TAxis.h"
#include "TBrowser.h"
#include "TVirtualPad.h"
#include "TPaveText.h"
#include "TGraphErrors.h"
#include "TStopwatch.h"
#include "TGaxis.h"
#include "TTimeStamp.h"
#include "TDatime.h"
#include "TMath.h"

ClassImp(TTreePerfStats)

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats() : TVirtualPerfStats()
{
   // default constructor (used when reading an object only)

   fName      = "";
   fHostInfo  = "";
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
   fUnzipTime     = 0;
   fCompress      = 0;
   fRealTimeAxis  = 0;
   fHostInfoText  = 0;
}

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats(const char *name, TTree *T) : TVirtualPerfStats()
{
   // Create a TTree I/O perf stats object.

   fName   = name;
   fTree   = T;
   T->SetPerfStats(this);
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
   fUnzipTime     = 0;
   fRealTimeAxis  = 0;
   fCompress      = (T->GetTotBytes()+0.00001)/T->GetZipBytes();

   Bool_t isUNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   if (isUNIX)
      fHostInfo = gSystem->GetFromPipe("uname -a");
   else
      fHostInfo = "Windows ";
   fHostInfo.Resize(20);
   fHostInfo += TString::Format("ROOT %s, Git: %s", gROOT->GetVersion(), gROOT->GetGitCommit());
   TDatime dt;
   fHostInfo += TString::Format(" %s",dt.AsString());
   fHostInfoText   = 0;

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
   delete fRealTimeAxis;
   delete fHostInfoText;

   if (gPerfStats == this) {
      gPerfStats = 0;
   }
}


//______________________________________________________________________________
void TTreePerfStats::Browse(TBrowser * /*b*/)
{
   // Browse

   Draw();
   gPad->Update();
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
   // on the real time axis ?
   distance = fRealTimeAxis->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {gPad->SetSelected(fRealTimeAxis);  return distance;}
   // on the host info label ?
   distance = fHostInfoText->DistancetoPrimitive(px,py);
   if (distance <kMaxDiff) {gPad->SetSelected(fHostInfoText);  return distance;}
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

   if (file == this->fFile){
      Long64_t offset = file->GetRelOffset();
      Int_t np = fGraphIO->GetN();
      Int_t entry = fTree->GetReadEntry();
      fGraphIO->SetPoint(np,entry,1e-6*offset);
      fGraphIO->SetPointError(np,0.001,1e-9*len);
      Double_t tnow = TTimeStamp();
      Double_t dtime = tnow-start;
      fDiskTime += dtime;
      fGraphTime->SetPoint(np,entry,tnow);
      fGraphTime->SetPointError(np,0.001,dtime);
      fReadCalls++;
      fBytesRead += len;
   }
}


//______________________________________________________________________________
void TTreePerfStats::UnzipEvent(TObject * tree, Long64_t /* pos */, Double_t start, Int_t /* complen */, Int_t /* objlen */)
{
   // Record TTree unzip event.
   // start is the TimeStamp before unzip
   // pos is where in the file the compressed buffer came from
   // complen is the length of the compressed buffer
   // objlen is the length of the de-compressed buffer

   if (tree == this->fTree){
      Double_t tnow = TTimeStamp();
      Double_t dtime = tnow-start;
      fUnzipTime += dtime;
   }
}

//______________________________________________________________________________
void TTreePerfStats::Finish()
{
   // When the run is finished this function must be called
   // to save the current parameters in the file and Tree in this object
   // the function is automatically called by Draw and Print

   if (fRealNorm)   return;  //has already been called
   if (!fFile)      return;
   if (!fTree)      return;
   fTreeCacheSize = fTree->GetCacheSize();
   fReadaheadSize = TFile::GetReadaheadSize();
   fBytesReadExtra= fFile->GetBytesReadExtra();
   fRealTime      = fWatch->RealTime();
   fCpuTime       = fWatch->CpuTime();
   Int_t npoints  = fGraphIO->GetN();
   if (!npoints) return;
   Double_t iomax = TMath::MaxElement(npoints,fGraphIO->GetY());
   fRealNorm      = iomax/fRealTime;
   fGraphTime->GetY()[0] = fRealNorm*fGraphTime->GetEY()[0];
   // we normalize the fGraphTime such that it can be drawn on top of fGraphIO
   for (Int_t i=1;i<npoints;i++) {
      fGraphTime->GetY()[i]   = fGraphTime->GetY()[i-1] +fRealNorm*fGraphTime->GetEY()[i];
      fGraphTime->GetEY()[i]  = 0;
   }
}


//______________________________________________________________________________
void TTreePerfStats::Paint(Option_t *option)
{
   // Draw the TTree I/O perf graph.

   Int_t npoints  = fGraphIO->GetN();
   if (!npoints) return;
   Double_t iomax = fGraphIO->GetY()[npoints-1];
   Double_t toffset=1;
   if (iomax >= 1e9) toffset = 1.2;
   fGraphIO->GetXaxis()->SetTitle("Tree entry number");
   fGraphIO->GetYaxis()->SetTitle("file position (MBytes)  ");
   fGraphIO->GetYaxis()->SetTitleOffset(toffset);
   fGraphIO->GetXaxis()->SetLabelSize(0.03);
   fGraphIO->GetYaxis()->SetLabelSize(0.03);
   fGraphIO->Paint(option);

   TString opts(option);
   opts.ToLower();
   Bool_t unzip = opts.Contains("unzip");

   //superimpose the time info (max 10 points)
   if (fGraphTime) {
      fGraphTime->Paint("l");
      TText tdisk(fGraphTime->GetX()[npoints-1],1.1*fGraphTime->GetY()[npoints-1],"RAW IO");
      tdisk.SetTextAlign(31);
      tdisk.SetTextSize(0.03);
      tdisk.SetTextColor(kRed);
      tdisk.Paint();
      if (!fRealTimeAxis) {
         Double_t uxmax = gPad->GetUxmax();
         Double_t uymax = gPad->GetUymax();
         Double_t rtmax = uymax/fRealNorm;
         fRealTimeAxis = new TGaxis(uxmax,0,uxmax,uymax,0.,rtmax,510,"+L");
         fRealTimeAxis->SetName("RealTimeAxis");
         fRealTimeAxis->SetLineColor(kRed);
         fRealTimeAxis->SetTitle("RealTime (s)  ");
         fRealTimeAxis->SetTitleColor(kRed);
         toffset = 1;
         if (fRealTime >=  100) toffset = 1.2;
         if (fRealTime >= 1000) toffset = 1.4;
         fRealTimeAxis->SetTitleOffset(toffset);
         fRealTimeAxis->SetLabelSize(0.03);
         fRealTimeAxis->SetLabelColor(kRed);
      }
      fRealTimeAxis->Paint();
   }

   Double_t extra = 100.*fBytesReadExtra/fBytesRead;
   if (!fPave) {
      fPave = new TPaveText(.01,.10,.24,.90,"brNDC");
      fPave->SetTextAlign(12);
      fPave->AddText(Form("TreeCache = %d MB",fTreeCacheSize/1000000));
      fPave->AddText(Form("N leaves  = %d",fNleaves));
      fPave->AddText(Form("ReadTotal = %g MB",1e-6*fBytesRead));
      fPave->AddText(Form("ReadUnZip = %g MB",1e-6*fBytesRead*fCompress));
      fPave->AddText(Form("ReadCalls = %d",fReadCalls));
      fPave->AddText(Form("ReadSize  = %7.3f KB",0.001*fBytesRead/fReadCalls));
      fPave->AddText(Form("Readahead = %d KB",fReadaheadSize/1000));
      fPave->AddText(Form("Readextra = %5.2f per cent",extra));
      fPave->AddText(Form("Real Time = %7.3f s",fRealTime));
      fPave->AddText(Form("CPU  Time = %7.3f s",fCpuTime));
      fPave->AddText(Form("Disk Time = %7.3f s",fDiskTime));
      if (unzip) {
         fPave->AddText(Form("UnzipTime = %7.3f s",fUnzipTime));
      }
      fPave->AddText(Form("Disk IO   = %7.3f MB/s",1e-6*fBytesRead/fDiskTime));
      fPave->AddText(Form("ReadUZRT  = %7.3f MB/s",1e-6*fCompress*fBytesRead/fRealTime));
      fPave->AddText(Form("ReadUZCP  = %7.3f MB/s",1e-6*fCompress*fBytesRead/fCpuTime));
      fPave->AddText(Form("ReadRT    = %7.3f MB/s",1e-6*fBytesRead/fRealTime));
      fPave->AddText(Form("ReadCP    = %7.3f MB/s",1e-6*fBytesRead/fCpuTime));
   }
   fPave->Paint();

   if (!fHostInfoText) {
      fHostInfoText = new TText(0.01,0.01,fHostInfo.Data());
      fHostInfoText->SetNDC();
      fHostInfoText->SetTextSize(0.025);
   }
   fHostInfoText->Paint();
}

//______________________________________________________________________________
void TTreePerfStats::Print(Option_t * option) const
{
   // Print the TTree I/O perf stats.

   TString opts(option);
   opts.ToLower();
   Bool_t unzip = opts.Contains("unzip");
   TTreePerfStats *ps = (TTreePerfStats*)this;
   ps->Finish();

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
   if (unzip) {
      printf("Strm Time = %7.3f seconds\n",fCpuTime-fUnzipTime);
      printf("UnzipTime = %7.3f seconds\n",fUnzipTime);
   }
   printf("Disk IO   = %7.3f MBytes/s\n",1e-6*fBytesRead/fDiskTime);
   printf("ReadUZRT  = %7.3f MBytes/s\n",1e-6*fCompress*fBytesRead/fRealTime);
   printf("ReadUZCP  = %7.3f MBytes/s\n",1e-6*fCompress*fBytesRead/fCpuTime);
   printf("ReadRT    = %7.3f MBytes/s\n",1e-6*fBytesRead/fRealTime);
   printf("ReadCP    = %7.3f MBytes/s\n",1e-6*fBytesRead/fCpuTime);
   if (unzip) {
      printf("ReadStrCP = %7.3f MBytes/s\n",1e-6*fCompress*fBytesRead/(fCpuTime-fUnzipTime));
      printf("ReadZipCP = %7.3f MBytes/s\n",1e-6*fCompress*fBytesRead/fUnzipTime);
   }
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
void TTreePerfStats::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<std::endl;
   if (gROOT->ClassSaved(TTreePerfStats::Class())) {
      out<<"   ";
   } else {
      out<<"   TTreePerfStats *";
   }
   out<<"ps = new TTreePerfStats();"<<std::endl;
   out<<"   ps->SetName("<<quote<<GetName()<<quote<<");"<<std::endl;
   out<<"   ps->SetHostInfo("<<quote<<GetHostInfo()<<quote<<");"<<std::endl;
   out<<"   ps->SetTreeCacheSize("<<fTreeCacheSize<<");"<<std::endl;
   out<<"   ps->SetNleaves("<<fNleaves<<");"<<std::endl;
   out<<"   ps->SetReadCalls("<<fReadCalls<<");"<<std::endl;
   out<<"   ps->SetReadaheadSize("<<fReadaheadSize<<");"<<std::endl;
   out<<"   ps->SetBytesRead("<<fBytesRead<<");"<<std::endl;
   out<<"   ps->SetBytesReadExtra("<<fBytesReadExtra<<");"<<std::endl;
   out<<"   ps->SetRealNorm("<<fRealNorm<<");"<<std::endl;
   out<<"   ps->SetRealTime("<<fRealTime<<");"<<std::endl;
   out<<"   ps->SetCpuTime("<<fCpuTime<<");"<<std::endl;
   out<<"   ps->SetDiskTime("<<fDiskTime<<");"<<std::endl;
   out<<"   ps->SetUnzipTime("<<fUnzipTime<<");"<<std::endl;
   out<<"   ps->SetCompress("<<fCompress<<");"<<std::endl;

   Int_t i, npoints = fGraphIO->GetN();
   out<<"   TGraphErrors *psGraphIO = new TGraphErrors("<<npoints<<");"<<std::endl;
   out<<"   psGraphIO->SetName("<<quote<<fGraphIO->GetName()<<quote<<");"<<std::endl;
   out<<"   psGraphIO->SetTitle("<<quote<<fGraphIO->GetTitle()<<quote<<");"<<std::endl;
   out<<"   ps->SetGraphIO(psGraphIO);"<<std::endl;
   fGraphIO->SaveFillAttributes(out,"psGraphIO",0,1001);
   fGraphIO->SaveLineAttributes(out,"psGraphIO",1,1,1);
   fGraphIO->SaveMarkerAttributes(out,"psGraphIO",1,1,1);
   for (i=0;i<npoints;i++) {
      out<<"   psGraphIO->SetPoint("<<i<<","<<fGraphIO->GetX()[i]<<","<<fGraphIO->GetY()[i]<<");"<<std::endl;
      out<<"   psGraphIO->SetPointError("<<i<<",0,"<<fGraphIO->GetEY()[i]<<");"<<std::endl;
   }
   npoints = fGraphTime->GetN();
   out<<"   TGraphErrors *psGraphTime = new TGraphErrors("<<npoints<<");"<<std::endl;
   out<<"   psGraphTime->SetName("<<quote<<fGraphTime->GetName()<<quote<<");"<<std::endl;
   out<<"   psGraphTime->SetTitle("<<quote<<fGraphTime->GetTitle()<<quote<<");"<<std::endl;
   out<<"   ps->SetGraphTime(psGraphTime);"<<std::endl;
   fGraphTime->SaveFillAttributes(out,"psGraphTime",0,1001);
   fGraphTime->SaveLineAttributes(out,"psGraphTime",1,1,1);
   fGraphTime->SaveMarkerAttributes(out,"psGraphTime",1,1,1);
   for (i=0;i<npoints;i++) {
      out<<"   psGraphTime->SetPoint("<<i<<","<<fGraphTime->GetX()[i]<<","<<fGraphTime->GetY()[i]<<");"<<std::endl;
      out<<"   psGraphTime->SetPointError("<<i<<",0,"<<fGraphTime->GetEY()[i]<<");"<<std::endl;
   }

   out<<"   ps->Draw("<<quote<<option<<quote<<");"<<std::endl;
}
