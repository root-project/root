// @(#)root/base:$Id$
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
#include "TFile.h"
#include "TTree.h"
#include "TAxis.h"
#include "TPaveText.h"
#include "TGraphErrors.h"
#include "TStopwatch.h"


ClassImp(TTreePerfStats)

//______________________________________________________________________________
TTreePerfStats::TTreePerfStats(TTree *T) : TVirtualPerfStats()
{
   // Return the name of the event type.
   
   fTree = T;
   fFile = T->GetCurrentFile();
   fGraph = new TGraphErrors(0);
   fGraph->SetName("ioperf");
   fGraph->SetTitle(Form("%s/%s",fFile->GetName(),T->GetName()));
   fGraph->SetUniqueID(999999999);
   fWatch  = new TStopwatch();
   fWatch->Start(); 
   gPerfStats = this;
}

//______________________________________________________________________________
void TTreePerfStats::Draw(Option_t * /*option*/)
{
   // Draw the graph
   
   fGraph->Draw("ap");
   fGraph->GetXaxis()->SetTitle("Tree entry number");
   fGraph->GetYaxis()->SetTitle("file position");
   Double_t extra = 100.*fFile->GetBytesReadExtra()/fFile->GetBytesRead();
   TPaveText *pv = new TPaveText(.2,.55,.5,.88,"brNDC");
   pv->SetTextAlign(12);
   pv->AddText(Form("TreeCache = %d MBytes",fTree->GetCacheSize()/1000000));
   pv->AddText(Form("ReadTotal = %g MBytes",1e-6*fFile->GetBytesRead()));
   pv->AddText(Form("ReadCalls = %d",fFile->GetReadCalls()));
   pv->AddText(Form("ReadSize  = %g KBytes/read",0.001*fFile->GetBytesRead()/fFile->GetReadCalls()));
   pv->AddText(Form("Readahead = %d KBytes",TFile::GetReadaheadSize()/1000));
   pv->AddText(Form("Readextra = %5.2f per cent",extra));
   pv->AddText(Form("Real Time = %7.3f seconds",fWatch->RealTime()));
   pv->AddText(Form("CPU  Time = %7.3f seconds",fWatch->CpuTime()));
   pv->AddText(Form("ReadRT    = %7.3f MBytes/s",1e-6*fFile->GetBytesRead()/fWatch->RealTime()));
   pv->AddText(Form("ReadCP    = %7.3f MBytes/s",1e-6*fFile->GetBytesRead()/fWatch->CpuTime()));
   pv->Draw();
}

//______________________________________________________________________________
void TTreePerfStats::FileReadEvent(TFile *file, Int_t len, Double_t /*proctime*/)
{
   // Return the name of the event type.
   
   Long64_t offset = file->GetRelOffset();
   Int_t np = fGraph->GetN();
   Int_t entry = fTree->GetReadEntry();
   Double_t err = len/2.;
   fGraph->SetPoint(np,entry,offset-err);
   fGraph->SetPointError(np,0.01,len);
}

//______________________________________________________________________________
void TTreePerfStats::Print(Option_t * /*option*/) const
{
   // Draw the graph
   
   Double_t extra = 100.*fFile->GetBytesReadExtra()/fFile->GetBytesRead();
   printf("TreeCache = %d MBytes\n",Int_t(fTree->GetCacheSize()/1000000));
   printf("ReadTotal = %g MBytes\n",1e-6*fFile->GetBytesRead());
   printf("ReadCalls = %d\n",fFile->GetReadCalls());
   printf("ReadSize  = %g KBytes/read\n",0.001*fFile->GetBytesRead()/fFile->GetReadCalls());
   printf("Readahead = %d KBytes\n",TFile::GetReadaheadSize()/1000);
   printf("Readextra = %5.2f per cent\n",extra);
   printf("Real Time = %7.3f seconds\n",fWatch->RealTime());
   printf("CPU  Time = %7.3f seconds\n",fWatch->CpuTime());
   printf("ReadRT    = %7.3f MBytes/s\n",1e-6*fFile->GetBytesRead()/fWatch->RealTime());
   printf("ReadCP    = %7.3f MBytes/s\n",1e-6*fFile->GetBytesRead()/fWatch->CpuTime());
}
