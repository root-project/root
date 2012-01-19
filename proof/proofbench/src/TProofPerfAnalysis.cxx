// @(#)root/proofx:$Id$
// Author: G.Ganis Nov 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPerfAnalysis                                                       //
//                                                                      //
// Set of tools to analyse the performance tree                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <errno.h>

#include "TProofPerfAnalysis.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "THashList.h"
#include "TKey.h"
#include "TList.h"
#include "TSortedList.h"
#include "TPerfStats.h"
#include "TRegexp.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

//
// Auxilliary internal class
//_______________________________________________________________________
class TProofPerfAnalysis::TWrkInfo : public TNamed {
public:
   TWrkInfo(const char *ord, const char *name) :
      TNamed(ord, name), fPackets(0), fRemotePackets(0), fEventsProcessed(0),
      fBytesRead(0), fLatency(0), fProcTime(0), fCpuTime(0), fStart(0), fStop(-1),
      fRateT(0), fRateRemoteT(0), fMBRateT(0), fMBRateRemoteT(0), fLatencyT(0) { }
   virtual ~TWrkInfo() { SafeDelete(fRateT); SafeDelete(fRateRemoteT);
                         SafeDelete(fMBRateT); SafeDelete(fMBRateRemoteT);
                         SafeDelete(fLatencyT); }

   Int_t     fPackets;          // Number of packets processed
   Int_t     fRemotePackets;    // Number of processed packet from non-local files
   Long64_t  fEventsProcessed;  // Tot events processed
   Long64_t  fBytesRead;        // Tot bytes read
   Double_t  fLatency;          // Tot latency
   Double_t  fProcTime;         // Tot proc time
   Double_t  fCpuTime;          // Tot CPU time

   Float_t   fStart;            // Start time
   Float_t   fStop;             // Stop time

   TGraph   *fRateT;             // Event processing rate vs time
   TGraph   *fRateRemoteT;       // Event processing rate of remote packets vs time
   TGraph   *fMBRateT;            // Byte processing rate vs time
   TGraph   *fMBRateRemoteT;      // Byte processing rate of remote packets vs time
   TGraph   *fLatencyT;          // Packet latency vs time

   Double_t  AvgRate() { if (fProcTime > 0) return (fEventsProcessed/fProcTime); return -1.; }
   Double_t  AvgIO() { if (fProcTime > 0) return (fBytesRead/fProcTime); return -1.; }

   void Print(Option_t * = "") const {
      Printf(" +++ TWrkInfo ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ");
      Printf(" +++ Worker:             %s, %s", GetName(), GetTitle());
      Printf(" +++ Activity interval:  %f -> %f", fStart, fStop);
      Printf(" +++ Amounts processed:  %d packets (%d remote), %lld evts, %lld bytes",
                                       fPackets, fRemotePackets, fEventsProcessed, fBytesRead);
      if (fProcTime) {
         Printf(" +++ Processing time:    %f s (CPU: %f s)", fProcTime, fCpuTime);
         Printf(" +++ Averages:           %f evts/s, %f MB/s", (Double_t)fEventsProcessed / fProcTime, (Double_t)fBytesRead /1024./1024./fProcTime);
      }
      Printf(" +++ Total latency:      %f", fLatency);
      Printf(" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ");
   }

   Int_t Compare(const TObject *o) const { TWrkInfo *wi = (TWrkInfo *)o;
                                           if (wi) {
                                             if (fStop < wi->fStop) {
                                                return -1;
                                             } else if (fStop == wi->fStop) {
                                                return 0;
                                             }
                                          }
                                          return 1; }
};

Int_t TProofPerfAnalysis::fgDebug = 0;
//________________________________________________________________________
TProofPerfAnalysis::TProofPerfAnalysis(const char *perffile,
                               const char *title, const char *treename)
               : TNamed(perffile, title), fTreeName(treename),
                 fInitTime(-1.), fMergeTime(-1.), fMaxTime(-1.),
                 fEvents(0), fPackets(0),
                 fEvtRateMax(-1.), fMBRateMax(-1.), fLatencyMax(-1.)
{
   // Constructor: open the file and attach to the tree

   // Use default title, if not specified
   if (!title) SetTitle("PROOF Performance Analysis");

   fTree = 0;
   fFile = TFile::Open(perffile);
   if (!fFile || (fFile && fFile->IsZombie())) {
      SafeDelete(fFile);
      Error("TProofPerfAnalysis", "problems opening file '%s'",
                              perffile ? perffile : "<undef>"); 
      return;
   }

   // Set the subdirectory name, if any
   if (fTreeName.Contains("/")) {
      fDirName = gSystem->DirName(fTreeName);
      fTreeName = gSystem->BaseName(fTreeName);
   }

   // Adjust the name, if requested
   if (fTreeName.BeginsWith("+"))
      fTreeName.Replace(0, 1, "PROOF_PerfStats");

   // Point to the right TDirectory
   TDirectory *dir = fFile;
   if (!fDirName.IsNull()) {
      if (!(dir = dynamic_cast<TDirectory *>(fFile->Get(fDirName)))) {
         Error("TProofPerfAnalysis", "directory '%s' not found or not loadable", fDirName.Data());
         fFile->Close();
         SafeDelete(fFile);
         return;
      }
   }
   
   // Load the performance tree
   LoadTree(dir);
   if (!fTree) {
      Error("TProofPerfAnalysis", "tree '%s' not found or not loadable", fTreeName.Data());
      fFile->Close();
      SafeDelete(fFile);
      return;
   }
   Printf(" +++ TTree '%s' has %lld entries", fTreeName.Data(), fTree->GetEntries());

   // Init worker information
   FillWrkInfo();
   
   // Done
   return;
}

//________________________________________________________________________
TProofPerfAnalysis::~TProofPerfAnalysis()
{
   // Destructor: detach the tree and close the file

   SafeDelete(fEvents);
   SafeDelete(fPackets);
   if (fFile) fFile->Close();
   SafeDelete(fFile);
}

//________________________________________________________________________
TString TProofPerfAnalysis::GetCanvasTitle(const char *t)
{
   // If defined, add '- <this title>' to the canvas title 't'

   if (fTitle.IsNull()) return TString(t);
   
   TString newt;
   if (t && strlen(t) > 0) {
      newt.Form("%s - %s", t, GetTitle());
   } else {
      newt = GetTitle();
   }
   // Done
   return newt;
}

//________________________________________________________________________
void TProofPerfAnalysis::LoadTree(TDirectory *dir)
{
   // Load tree fTreeName from directory 'dir'. If not found, look for the
   // first TTree in the directory (and sub-directories) with the name containing
   // fTreeName.
   // The tree pointer is saved in fTree.

   fTree = 0;
   if (!dir) return;
   
   // Try first the full name in the top directory
   if ((fTree = dynamic_cast<TTree *>(dir->Get(fTreeName)))) return;
   
   TRegexp re(fTreeName);
   // Now look inside: iter on the list of keys first
   TIter nxk(dir->GetListOfKeys());
   TKey *k = 0;
   while ((k = (TKey *) nxk())) {
      if (!strcmp(k->GetClassName(), "TDirectoryFile")) {
         TDirectory *kdir = (TDirectory *) dir->Get(k->GetName());
         LoadTree(kdir);
         if (fTree) return;
      } else if (!strcmp(k->GetClassName(), "TTree")) {
         TString tn(k->GetName());
         if (tn.Index(re) != kNPOS) {
            if ((fTree = dynamic_cast<TTree *>(dir->Get(tn)))) {
               fTreeName = tn;
               Printf(" +++ Found and loaded TTree '%s'", tn.Data());
               return;
            }
         }
      }
   }
   
   // Nothing found
   return;
}

//________________________________________________________________________
void TProofPerfAnalysis::FileDist(Bool_t writedet)
{
   // Analyse the file distribution. If writedet, underling details are 
   // written out to a text file.

   if (!IsValid()) {
      Error("FileDist","not a valid instance - do nothing");
      return;
   }

   // Fill file info
   TList *wrkList = new TList;
   TList *srvList = new TList;
   GetFileInfo(wrkList, srvList);
   Info("FileDist", "%d workers were active during this query", wrkList->GetSize());
   Info("FileDist", "%d servers were active during this query", srvList->GetSize());

   // Fill the worker-data server mapping
   TIter nxs(srvList);
   TIter nxw(wrkList);
   TNamed *sn = 0, *wn = 0;
   while ((sn = (TNamed *)nxs())) {
      nxw.Reset();
      while ((wn = (TNamed *) nxw())) {
         if (!strcmp(TUrl(sn->GetName()).GetHostFQDN(), wn->GetTitle())) {
            sn->SetTitle(wn->GetName());
         }
      }
   }

   // Reorder the lists following the title
   TList *nwl = new TList;
   TList *nsl = new TList;
   nxw.Reset();
   while ((wn = (TNamed *) nxw())) {
      TIter nnxw(nwl);
      TNamed *nwn = 0;
      while ((nwn = (TNamed *) nnxw())) {
         if (CompareOrd(wn->GetName(), nwn->GetName()) < 0) {
            nwl->AddBefore(nwn, wn);
            break;
         }
      }
      if (!nwn) nwl->Add(wn);
      // Find the server name, if any
      nxs.Reset();
      while ((sn = (TNamed *)nxs())) {
         if (!strcmp(sn->GetTitle(), wn->GetName())) {
            TIter nnxs(nsl);
            TNamed *nsn = 0;
            while ((nsn = (TNamed *) nnxs())) {
               if (CompareOrd(sn->GetTitle(), nsn->GetTitle()) < 0) {
                  nsl->AddBefore(nsn, sn);
                  break;
               }
            }
            if (!nsn) nsl->Add(sn);
            break;
         }
      }
      if (sn) srvList->Remove(sn);
   }
   // Add remaining servers at the end
   nxs.Reset();
   while ((sn = (TNamed *)nxs())) {
      nsl->Add(sn);
   }
   // Clean the orginal lists
   wrkList->SetOwner(kFALSE);
   srvList->SetOwner(kFALSE);
   delete wrkList;
   delete srvList;
   wrkList = nwl;
   srvList = nsl;

   // Notify
   wrkList->ls();
   srvList->ls();

   // Separate out the case with only one file server
   if (srvList->GetSize() == 1) {

      Printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ");
      Printf(" + Only one data server found: full analysis meaningful  + ");
      Printf(" + only when there are more file servers                 + ");
      Printf(" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n");
      
      
      // Create a 1D histo for cross packets
      TH1F *hxpak = new TH1F("hxpak", "MBytes / Worker",
                                    wrkList->GetSize(), 0., (Double_t)wrkList->GetSize());
      hxpak->SetDirectory(0);
      hxpak->SetMinimum(0.);
      hxpak->GetXaxis()->SetTitle("Worker");

      // Set the labels
      Int_t j = 1;
      TIter nxsw(wrkList);
      while ((wn = (TNamed *)nxsw())) {
         hxpak->GetXaxis()->SetBinLabel(j++, wn->GetName());
      }

      // Fill the histograms
      FillFileDistOneSrv(hxpak, writedet);

      // Display histos
      gStyle->SetOptStat(0);

      TCanvas *c2 = new TCanvas("cv-hxpak",  GetCanvasTitle(hxpak->GetTitle()), 800,350,700,700);
      c2->cd();
      hxpak->Draw();
      c2->Update();
   
   } else {
      // Create a 1D histo for file distribution
      TH1F *hfdis = new TH1F("hfdis", "Packet dist",
                           srvList->GetSize(), 0., (Double_t)srvList->GetSize());
      hfdis->SetDirectory(0);
      hfdis->SetMinimum(0);
      hfdis->GetXaxis()->SetTitle("Server");
      TH1F *hbdis = new TH1F("hbdis", "MBytes dist",
                           srvList->GetSize(), 0., (Double_t)srvList->GetSize());
      hbdis->SetDirectory(0);
      hbdis->SetMinimum(0);
      hbdis->GetXaxis()->SetTitle("Server");
      // Create a 2D histo for cross packets
      TH2F *hxpak = new TH2F("hxpak", "MBytes / {Worker,Server}",
                                    srvList->GetSize(), 0., (Double_t)srvList->GetSize(),
                                    wrkList->GetSize(), 0., (Double_t)wrkList->GetSize());
      hxpak->SetDirectory(0);
      hxpak->GetYaxis()->SetTitle("Worker");
      hxpak->GetXaxis()->SetTitle("Server");
      hxpak->GetXaxis()->SetTitleOffset(1.4);
      hxpak->GetYaxis()->SetTitleOffset(1.7);

      // Set the labels
      Int_t j = 1;
      TIter nxsw(wrkList);
      while ((wn = (TNamed *)nxsw())) {
         hxpak->GetYaxis()->SetBinLabel(j++, wn->GetName());
      }
      j = 1;
      TIter nxss(srvList);
      while ((sn = (TNamed *)nxss())) {
         hfdis->GetXaxis()->SetBinLabel(j, sn->GetName());
         hbdis->GetXaxis()->SetBinLabel(j, sn->GetName());
         hxpak->GetXaxis()->SetBinLabel(j++, sn->GetName());
      }

      // Fill the histograms
      FillFileDist(hfdis, hbdis, hxpak, writedet);

      j = 1;
      nxss.Reset();
      while ((sn = (TNamed *)nxss())) {
         TString lab(sn->GetName());
         lab = TUrl(sn->GetName()).GetHost();
         if (strcmp(sn->GetTitle(), "remote") && lab.Index(".") != kNPOS) lab.Remove(lab.Index("."));
         hfdis->GetXaxis()->SetBinLabel(j, lab);
         hbdis->GetXaxis()->SetBinLabel(j, lab);
         hxpak->GetXaxis()->SetBinLabel(j++, lab);
      }

      // Display histos
      gStyle->SetOptStat(0);

      TCanvas *c1 = new TCanvas("cv-hfdis", GetCanvasTitle(hfdis->GetTitle()), 800,50,700,700);
      c1->Divide(1,2);
      TPad *pad1 = (TPad *) c1->GetPad(1);
      TPad *pad2 = (TPad *) c1->GetPad(2);
      pad1->cd();
      hfdis->Draw();
      pad2->cd();
      hbdis->Draw();
      c1->Update();

      TCanvas *c2 = new TCanvas("cv-hxpak",  GetCanvasTitle(hxpak->GetTitle()), 500,350,700,700);
      c2->cd();
      hxpak->Draw("lego");
      c2->Update();
   }
   // Done
   return;
}

//________________________________________________________________________
void TProofPerfAnalysis::GetFileInfo(TList *wl, TList *sl)
{
   // Fill file info

   if (!wl || !sl) return;

   // Extract information
   TPerfEvent pe;
   TPerfEvent* pep = &pe;
   fTree->SetBranchAddress("PerfEvents", &pep);
   Long64_t entries = fTree->GetEntries();
   TNamed *wn = 0, *sn = 0;
   for (Long64_t k=0; k<entries; k++) {
      fTree->GetEntry(k);
      // Analyse only packets
      if (pe.fType != TVirtualPerfStats::kPacket) continue;
      // Find out the worker instance
      TString wrk(TUrl(pe.fSlaveName.Data()).GetHostFQDN());
      wn = (TNamed *) wl->FindObject(pe.fSlave.Data());
      if (!wn) {
         wn = new TNamed(pe.fSlave.Data(), wrk.Data());
         wl->Add(wn);
      }
      // Find out the file server instance
      TUrl uf(pe.fFileName);
      TString srv(uf.GetUrl());
      Int_t ifn = srv.Index(uf.GetFile());
      if (ifn != kNPOS) srv.Remove(ifn);
      sn = (TNamed *) sl->FindObject(srv.Data());
      if (!sn) {
         sn = new TNamed(srv.Data(), "remote");
         sl->Add(sn);
      }
   }

   // Done
   return;
}

//________________________________________________________________________
Int_t TProofPerfAnalysis::CompareOrd(const char *ord1, const char *ord2)
{
   // Return -1 if ord1 comes before ord2, 0 i they are equal,
   // 1 if ord1 comes after ord2

   TString o1(ord1), o2(ord2), p1, p2;
   Int_t o1d = 0, o2d = 0;
   if ((o1d = o1.CountChar('.')) > (o2d = o2.CountChar('.'))) {
      return 1;
   } else if (o1d < o2d) {
      return -1;
   } else {
      o1.ReplaceAll(".", " ");
      o2.ReplaceAll(".", " ");
      Bool_t b1 = o1.Tokenize(p1, o1d, " ");
      Bool_t b2 = o2.Tokenize(p2, o2d, " ");
      while (b1 && b2) {
         if (p1.Atoi() > p2.Atoi()) {
            return 1;
         } else if (p1.Atoi() < p2.Atoi()) {
            return -1;
         } else {
            b1 = o1.Tokenize(p1, o1d, " ");
            b2 = o2.Tokenize(p2, o2d, " ");
         }
      }
      if (b1 && !b2) {
         return 1;
      } else if (b2 && !b1) {
         return -1;
      } else {
         return 0;
      }
   }
}

//________________________________________________________________________
void TProofPerfAnalysis::FillFileDist(TH1F *hf, TH1F *hb, TH2F *hx, Bool_t wdet)
{
   // Fill file info

   if (!hf || !hb || !hx) return;

   TString fnout;
   FILE *fout = 0;
   if (wdet) {
      fnout.Form("%s-FileDist-Details.txt", GetName());
      if (!(fout = fopen(fnout.Data(), "w"))) {
         Warning("FillFileDist", "asked to save details in '%s' but file could"
                                 " not be open (errno: %d)", fnout.Data(), (int)errno);
      } else {
         Info("FillFileDist", "saving details to '%s'", fnout.Data());
      }
   }
   // Extract information
   TPerfEvent pe;
   TPerfEvent* pep = &pe;
   fTree->SetBranchAddress("PerfEvents",&pep);
   Long64_t entries = fTree->GetEntries();
   for (Long64_t k=0; k<entries; k++) {
      fTree->GetEntry(k);
      // Analyse only packets
      if (pe.fType != TVirtualPerfStats::kPacket) continue;
      // Find out the labels ...
      TString wrk(pe.fSlave.Data());
      TUrl uf(pe.fFileName);
      TString srv(uf.GetUrl());
      Int_t ifn = srv.Index(uf.GetFile());
      if (ifn != kNPOS) srv.Remove(ifn);
      // ... and the bins
      Double_t xhf = hf->GetXaxis()->GetBinCenter(hf->GetXaxis()->FindBin(srv.Data()));
      Double_t xhx = hx->GetXaxis()->GetBinCenter(hx->GetXaxis()->FindBin(srv.Data()));
      Double_t yhx = hx->GetYaxis()->GetBinCenter(hx->GetYaxis()->FindBin(wrk.Data()));
      // Save details, if asked
      if (fout)
         fprintf(fout, "%s,%s -> %f,%f (%f)\n",
                       srv.Data(), wrk.Data(), xhx, yhx, pe.fBytesRead / 1024.);
      // Fill now
      hf->Fill(xhf);
      hb->Fill(xhf, pe.fBytesRead / 1024. / 1024.);
      hx->Fill(xhx, yhx, pe.fBytesRead / 1024. / 1024.);
   }
   if (fout) fclose(fout);
   // Done
   return;
}

//________________________________________________________________________
void TProofPerfAnalysis::FillFileDistOneSrv(TH1F *hx, Bool_t wdet)
{
   // Fill file info when there is only one file server

   if (!hx) return;

   TString fnout;
   FILE *fout = 0;
   if (wdet) {
      fnout.Form("%s-FileDist-Details.txt", GetName());
      if (!(fout = fopen(fnout.Data(), "w"))) {
         Warning("FillFileDistOneSrv", "asked to save details in '%s' but file could"
                                       " not be open (errno: %d)", fnout.Data(), (int)errno);
      } else {
         Info("FillFileDistOneSrv", "saving details to '%s'", fnout.Data());
      }
   }
   // Extract information
   TPerfEvent pe;
   TPerfEvent* pep = &pe;
   fTree->SetBranchAddress("PerfEvents",&pep);
   Long64_t entries = fTree->GetEntries();
   for (Long64_t k=0; k<entries; k++) {
      fTree->GetEntry(k);
      // Analyse only packets
      if (pe.fType != TVirtualPerfStats::kPacket) continue;
      // Find out the labels ...
      TString wrk(pe.fSlave.Data());
      TUrl uf(pe.fFileName);
      TString srv(uf.GetUrl());
      Int_t ifn = srv.Index(uf.GetFile());
      if (ifn != kNPOS) srv.Remove(ifn);
      // ... and the bins
      Double_t xhx = hx->GetXaxis()->GetBinCenter(hx->GetXaxis()->FindBin(wrk.Data()));
      // Save details, if asked
      if (fout)
         fprintf(fout, "%s,%s -> %f (%f)\n",
                       srv.Data(), wrk.Data(), xhx, pe.fBytesRead / 1024.);
      // Fill now
      hx->Fill(xhx, pe.fBytesRead / 1024. / 1024.);
   }
   if (fout) fclose(fout);
   // Done
   return;
}

//________________________________________________________________________
void TProofPerfAnalysis::WorkerActivity()
{
   // Measure the worker activity

   if (!IsValid()) {
      Error("WorkerActivity","not a valid instance - do nothing");
      return;
   }

   // Fill basic worker info
   if (!WrkInfoOK()) FillWrkInfo();
   if (!WrkInfoOK()) {
      Error("WorkerActivity", "workers information not available - do nothing");
      return;
   }

   TObject *o = 0;
   // Create the histograms with activity vs time
   if ((o = gDirectory->FindObject("act10"))) delete o;
   Float_t t0 = fMergeTime - 2.* (fMaxTime - fMergeTime);
   Float_t t1 = 2.*fInitTime;
   if (t1 > t0) t1 = t0;
   TH1F *hact10 = new TH1F("act10", "Worker activity start (seconds)", 50, 0., t1);
   hact10->GetXaxis()->SetTitle("Query Processing Time (s)");
   if ((o = gDirectory->FindObject("act11"))) delete o;
   TH1F *hact11 = new TH1F("act11", "Worker activity stop (seconds)", 50, t0, fMaxTime);
   hact11->GetXaxis()->SetTitle("Query Processing Time (s)");
   if ((o = gDirectory->FindObject("act2"))) delete o;
   TH1F *hact2 = new TH1F("act2", "End of activity (seconds)", 50, t0, fMaxTime);
   hact2->GetXaxis()->SetTitle("Query Processing Time (s)");

   // Fine-tune stat printing
   Int_t curoptstat = gStyle->GetOptStat();
   gStyle->SetOptStat(1100);
   
   // Create the sorted list
   TIter nxw(&fWrksInfo);
   TWrkInfo *wi = 0;
   while ((wi = (TWrkInfo *)nxw())) {
      Int_t j = 0;
      for (j = 1; j < hact10->GetNbinsX()+1 ; j++) { 
         if (wi->fStart < hact10->GetBinLowEdge(j))
            hact10->Fill(hact10->GetBinCenter(j));
      }
      for (j = 1; j < hact11->GetNbinsX()+1 ; j++) { 
         if (wi->fStop > hact11->GetBinLowEdge(j))
            hact11->Fill(hact11->GetBinCenter(j));
      }
      hact2->Fill(wi->fStop);
   }

   // Display histos
   TCanvas *c1 = new TCanvas("perf", GetCanvasTitle("Activity histos"), 800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   pad1->Divide(2,1);
   TPad *pad10 = (TPad *) pad1->GetPad(1);
   TPad *pad11 = (TPad *) pad1->GetPad(2);
   pad10->cd();
   hact10->Draw();
   pad11->cd();
   hact11->Draw();
   TPad *pad2 = (TPad *) c1->GetPad(2);
   pad2->cd();
   hact2->Draw();
   c1->cd();
   c1->Update();

   // Restore stat options
   gStyle->SetOptStat(curoptstat);

   // Done
   return;
}

//________________________________________________________________________
void TProofPerfAnalysis::PrintWrkInfo(Int_t showlast)
{
   // Print information for all or the slowest showlast workers.
   // Use showlast < 0 to print all
   
   // Create the sorted list
   Int_t k = fWrksInfo.GetSize();
   TIter nxw(&fWrksInfo);
   TWrkInfo *wi = 0;
   while ((wi = (TWrkInfo *)nxw())) {
      // Print info about slowest workers
      k--;
      if (showlast < 0 || k < showlast) wi->Print();
   }
}

//________________________________________________________________________
void TProofPerfAnalysis::PrintWrkInfo(const char *wn)
{
   // Print information for worker 'wn' (ordinal) or on the machine whose
   // ordinal or fqdn matches 'wn'. Multiple specifications separated by ','
   // or ' ' are supported, as well as wildcards '*', e.g. '0.2*,lxb10* lxf2323.doma.in"

   if (!wn || (wn && strlen(wn) <= 0)) {
      Error("PrintWrkInfo", "worker name or host must be defined!");
      return;
   }

   // Check exact name
   TWrkInfo *wi = (TWrkInfo *) fWrksInfo.FindObject(wn);
   if (wi) {
      wi->Print();
   } else {
      // Check matching
      TString ww(wn), w;
      TIter nxw(&fWrksInfo);
      while ((wi = (TWrkInfo *)nxw())) {
         TString n(wi->GetName()), t(wi->GetTitle());
         Ssiz_t from = 0;
         while (ww.Tokenize(w, from, "[, ]")) {
            TRegexp re(w, kTRUE);
            if (n.Index(re) != kNPOS || t.Index(re) != kNPOS) wi->Print();
         }
      }
   }
}

//________________________________________________________________________
void TProofPerfAnalysis::FillWrkInfo(Bool_t force)
{
   // Fill basic worker info; if 'force' rescan the TTree even already done

   // Nothing to do if already called
   if (fWrksInfo.GetSize() > 0 && !force) return;

   // Cleanup existing information
   fWrksInfo.SetOwner(kTRUE);
   fWrksInfo.Clear();
   fInitTime = -1.;
   fMergeTime = -1.;
   fMaxTime = -1.;
   fEvtRateMax = -1.;
   fMBRateMax = -1.;
   fLatencyMax = -1.;
   
   TList *wl = new TList;
   // Extract worker information
   TPerfEvent pe;
   TPerfEvent* pep = &pe;
   fTree->SetBranchAddress("PerfEvents",&pep);
   Long64_t entries = fTree->GetEntries();
   TWrkInfo *wi = 0;
   for (Long64_t k=0; k<entries; k++) {
      fTree->GetEntry(k);
      // Analyse only packets
      if (pe.fType == TVirtualPerfStats::kPacket) {
         // Find out the worker instance
         wi = (TWrkInfo *) wl->FindObject(pe.fSlave.Data());
         if (!wi) {
            wi = new TWrkInfo(pe.fSlave.Data(), pe.fSlaveName.Data());
            wl->Add(wi);
            wi->fRateT = new TGraph(100);
            wi->fRateRemoteT = new TGraph(100);
            wi->fMBRateT = new TGraph(100);
            wi->fMBRateRemoteT = new TGraph(100);
            wi->fLatencyT = new TGraph(100);
         }
         // Add Info now
         if (wi->fPackets <= 0) {
            wi->fStart = pe.fTimeStamp.GetSec() + 1e-9*pe.fTimeStamp.GetNanoSec() - pe.fProcTime;
         } else {
            wi->fStop = pe.fTimeStamp.GetSec() + 1e-9*pe.fTimeStamp.GetNanoSec();
         }
         TUrl uf(pe.fFileName), uw(pe.fSlaveName);
         fMaxTime = pe.fTimeStamp.GetSec() + 1e-9*pe.fTimeStamp.GetNanoSec();
         wi->fEventsProcessed += pe.fEventsProcessed;
         wi->fBytesRead += pe.fBytesRead;
         wi->fLatency += pe.fLatency;
         wi->fProcTime += pe.fProcTime;
         wi->fCpuTime += pe.fCpuTime;
         // Fill graphs
         Double_t tt = pe.fTimeStamp.GetSec() + 1e-9*pe.fTimeStamp.GetNanoSec();
         Double_t ert = pe.fEventsProcessed / pe.fProcTime ;
         Double_t brt = pe.fBytesRead / pe.fProcTime / 1024. / 1024. ;
         wi->fRateT->SetPoint(wi->fPackets, tt, ert);
         if (brt > 0.) wi->fMBRateT->SetPoint(wi->fPackets, tt, brt);
         wi->fLatencyT->SetPoint(wi->fPackets, tt, pe.fLatency);
         if (!pe.fFileName.IsNull() && strcmp(uf.GetHostFQDN(), uw.GetHostFQDN())) {
            wi->fRateRemoteT->SetPoint(wi->fRemotePackets, tt, ert);
            wi->fMBRateRemoteT->SetPoint(wi->fRemotePackets, tt, brt);
            wi->fRemotePackets++;
         }
         wi->fPackets++;
         if (ert > fEvtRateMax) fEvtRateMax = ert;
         if (brt > fMBRateMax) fMBRateMax = brt;
         if (pe.fLatency > fLatencyMax) fLatencyMax = pe.fLatency;
         // Notify
         if (fgDebug > 1) {
            if (pe.fProcTime > 0.) {
               Printf(" +++ %s #:%d at:%fs lat:%fs proc:%fs evts:%lld bytes:%lld (rates:%f evt/s, %f MB/s)",
                     wi->GetName(), wi->fPackets, fMaxTime - pe.fProcTime,
                     pe.fLatency, pe.fProcTime, pe.fEventsProcessed, pe.fBytesRead,
                     ert, brt);
            } else {
               Printf(" +++ %s #:%d at:%fs lat:%fs proc:%fs rate:-- evt/s (-- bytes/s)",
                     wi->GetName(), wi->fPackets, fMaxTime, pe.fLatency, pe.fProcTime);
            }
         }
      } else if (pe.fType == TVirtualPerfStats::kStart) {
         Float_t start = pe.fTimeStamp.GetSec() + 1e-9*pe.fTimeStamp.GetNanoSec();
         if (fgDebug > 1) Printf(" +++ %s Start: %f s", pe.fEvtNode.Data(), start);
      } else if (pe.fType == TVirtualPerfStats::kStop) {
         Float_t stop = pe.fTimeStamp.GetSec() + 1e-9*pe.fTimeStamp.GetNanoSec();
         if (fgDebug > 1) Printf(" +++ %s Stop: %f s", pe.fEvtNode.Data(), stop);
      } else {
         if (fgDebug > 2) Printf(" +++ Event type: %d", pe.fType);
      }
   }
   // Final analysis to find relevant times
   fMergeTime = fMaxTime;
   TIter nxw(wl);
   while ((wi = (TWrkInfo *) nxw())) {
      fWrksInfo.Add(wi);
      if (wi->fStart > fInitTime) fInitTime = wi->fStart;
      if (wi->fStop < fMergeTime || fMergeTime < 0.) fMergeTime = wi->fStop;
      // Resize the graphs
      wi->fRateT->Set(wi->fPackets);
      wi->fRateRemoteT->Set(wi->fRemotePackets);
      wi->fLatencyT->Set(wi->fPackets);
      wi->fMBRateT->Set(wi->fPackets);
      wi->fMBRateRemoteT->Set(wi->fRemotePackets);
   }
   wl->SetOwner(kFALSE);
   delete wl;

   // (Re-)create the event and packet distribution histograms
   SafeDelete(fEvents);
   SafeDelete(fPackets);
   fEvents = new TH1F("hevents", "Events per worker", fWrksInfo.GetSize(), -.5, fWrksInfo.GetSize()-.5);
   fEvents->SetDirectory(0);
   fPackets = new TH1F("hpackets", "Packets per worker", fWrksInfo.GetSize(), -.5, fWrksInfo.GetSize()-.5);
   fPackets->SetDirectory(0);
   Int_t j = 0;
   TIter nxwi(&fWrksInfo);   
   while ((wi = (TWrkInfo *)nxwi())) {
      fEvents->GetXaxis()->SetBinLabel(j+1, wi->GetName());
      fEvents->Fill(j, wi->fEventsProcessed);
      fPackets->GetXaxis()->SetBinLabel(j+1, wi->GetName());
      fPackets->Fill(j++, wi->fPackets);
   }
   fEvents->SetMinimum(0.);
   fPackets->SetMinimum(0.);
   fEvents->SetFillColor(38);
   fPackets->SetFillColor(38);
   fEvents->GetYaxis()->SetTitle("Events");
   fEvents->GetXaxis()->SetTitle("Worker");
   fPackets->GetYaxis()->SetTitle("Packets");
   fPackets->GetXaxis()->SetTitle("Worker");
   
   // Print summary
   Printf(" +++ %d workers were active during this query", fWrksInfo.GetSize());
   Printf(" +++ Total query time: %f secs (init: %f secs, merge: %f secs)",
          fMaxTime, fInitTime, fMaxTime - fMergeTime);
}

//________________________________________________________________________
void TProofPerfAnalysis::SetDebug(Int_t d)
{
   // Static setter for the verbosity level
   
   fgDebug = d;
}

//________________________________________________________________________
void TProofPerfAnalysis::EventDist()
{
   // Display event and packet distribution

   if (!fEvents || !fPackets) {
      Error("EventDist", "distributions not initialized - do nothing");
   }

   // Display histos
   TCanvas *c1 = new TCanvas("evtdist", GetCanvasTitle("Event distributions"),800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   pad1->cd();
   fEvents->SetStats(kFALSE);
   fEvents->Draw();
   TPad *pad2 = (TPad *) c1->GetPad(2);
   pad2->cd();
   fPackets->SetStats(kFALSE);
   fPackets->Draw();
   c1->cd();
   c1->Update();

}

//________________________________________________________________________
void TProofPerfAnalysis::RatePlot(const char *wrks)
{
   // Show event processing or MB processing rate plot vs time

   // Create the histograms
   TObject *o = 0;
   if ((o = gDirectory->FindObject("rt1"))) delete o;
   TH1F *hrt1 = new TH1F("rt1", "Evt processing rate (evt/s)", 100, 0., fMaxTime);
   hrt1->SetMinimum(0.);
   hrt1->SetMaximum(1.05*fEvtRateMax);
   hrt1->SetStats(kFALSE);
   hrt1->GetXaxis()->SetTitle("Query Processing Time (s)");
   if ((o = gDirectory->FindObject("rt2"))) delete o;
   TH1F *hrt2 = new TH1F("rt2", "MB processing rate (MB/s)", 100, 0., fMaxTime);
   hrt2->SetMinimum(0.);
   hrt2->SetMaximum(1.05*fMBRateMax);
   hrt2->SetStats(kFALSE);
   hrt2->GetXaxis()->SetTitle("Query Processing Time (s)");

   // Display histo frames
   TCanvas *c1 = new TCanvas("rates", GetCanvasTitle("Processing rates"), 800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   pad1->cd();
   hrt1->Draw();
   TPad *pad2 = (TPad *) c1->GetPad(2);
   pad2->cd();
   hrt2->Draw();
   c1->cd();
   c1->Update();

   // Which workers?
   THashList *wl = 0;
   TString ww(wrks);
   if (!ww.IsNull() && ww != "*" && ww != "all") {
      TString w;
      Ssiz_t from = 0;
      while ((ww.Tokenize(w, from, ","))) {
         if (!wl) wl = new THashList();
         wl->Add(new TObjString(w.Data()));
      }
   }
      
   // Now plot the graphs per worker
   TIter nxw(&fWrksInfo);
   TWrkInfo *wi = 0;
   while ((wi = (TWrkInfo *) nxw())) {
      if (wl && !wl->FindObject(wi->GetName())) continue;
      if (wi->fRateT && wi->fRateT->GetN() > 0) {
         wi->fRateT->SetNameTitle(wi->GetName(), wi->GetTitle());
         pad1->cd();
         wi->fRateT->Draw("L");
      }
      if (wi->fRateRemoteT && wi->fRateRemoteT->GetN() > 0) {
         wi->fRateRemoteT->SetNameTitle(wi->GetName(), wi->GetTitle());
         pad1->cd();
         wi->fRateRemoteT->SetLineColor(kRed);
         wi->fRateRemoteT->Draw("L");
      }
      if (wi->fMBRateT && wi->fMBRateT->GetN() > 0) {
         wi->fMBRateT->SetNameTitle(wi->GetName(), wi->GetTitle());
         pad2->cd();
         wi->fMBRateT->Draw("L");
      }
      if (wi->fMBRateRemoteT && wi->fMBRateRemoteT->GetN() > 0) {
         wi->fMBRateRemoteT->SetNameTitle(wi->GetName(), wi->GetTitle());
         pad2->cd();
         wi->fMBRateRemoteT->SetLineColor(kRed);
         wi->fMBRateRemoteT->Draw("L");
      }
      c1->cd();
      c1->Update();
   }

   // Cleanup
   if (wl) {
      wl->SetOwner(kTRUE);
      delete wl;
   }
}

//________________________________________________________________________
void TProofPerfAnalysis::LatencyPlot(const char *wrks)
{
   // Show event processing or MB processing rate plot vs time
   // Create the histograms

   TObject *o = 0;
   if ((o = gDirectory->FindObject("lt1"))) delete o;
   TH1F *hlt1 = new TH1F("lt1", "Packet retrieval latency", 100, 0., fMaxTime);
   hlt1->SetMinimum(0.);
   hlt1->SetMaximum(1.05*fLatencyMax);
   hlt1->SetStats(kFALSE);
   hlt1->GetXaxis()->SetTitle("Query Processing Time (s)");
   hlt1->GetYaxis()->SetTitle("Latency (s)");
   
   // Display histo frames
   TCanvas *c1 = new TCanvas("latency", GetCanvasTitle("Packet Retrieval Latency"), 800,10,700,780);
   hlt1->Draw();
   c1->cd();
   c1->Update();

   // Which workers?
   THashList *wl = 0;
   TString ww(wrks);
   if (!ww.IsNull() && ww != "*" && ww != "all") {
      TString w;
      Ssiz_t from = 0;
      while ((ww.Tokenize(w, from, ","))) {
         if (!wl) wl = new THashList();
         wl->Add(new TObjString(w.Data()));
      }
   }
      
   // Now plot the graphs per worker
   TIter nxw(&fWrksInfo);
   TWrkInfo *wi = 0;
   while ((wi = (TWrkInfo *) nxw())) {
      if (wl && !wl->FindObject(wi->GetName())) continue;
      if (wi->fLatencyT) {
         wi->fLatencyT->SetNameTitle(wi->GetName(), wi->GetTitle());
         wi->fLatencyT->Draw("L");
      }
      c1->cd();
      c1->Update();
   }

   // Cleanup
   if (wl) {
      wl->SetOwner(kTRUE);
      delete wl;
   }
}
