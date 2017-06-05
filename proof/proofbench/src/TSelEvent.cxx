// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSelEvent
\ingroup proofbench

Selector for PROOF I/O benchmark test.
For the I/O benchmark, event files are read in and histograms are filled.
For memory clean-up, dedicated files large enough to clean up memory 
cache on the machine are read in. Or memory clean-up can be 
accompolished by system call on Linux machine inside SlaveBegin(..) 
which should be much faster the reading in large files.

*/

#define TSelEvent_cxx

#include "TSelEvent.h"
#include <TH1F.h>
#include <TStyle.h>
#include "TParameter.h"
#include "TProofBenchTypes.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TFileInfo.h"
#include "THashList.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "TSystem.h"
#include "TROOT.h"

ClassImp(TSelEvent);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TSelEvent::TSelEvent(TTree *)
          : fReadType(0), fDebug(kFALSE), fCHist(0), fPtHist(0),
            fNTracksHist(0), fEventName(0), fTracks(0), fHighPt(0), fMuons(0),
            fH(0), b_event_fType(0), b_fEventName(0), b_event_fNtrack(0), b_event_fNseg(0),
            b_event_fNvertex(0), b_event_fFlag(0), b_event_fTemperature(0),
            b_event_fMeasures(0), b_event_fMatrix(0), b_fClosestDistance(0),
            b_event_fEvtHdr(0), b_fTracks(0), b_fHighPt(0), b_fMuons(0),
            b_event_fLastTrack(0), b_event_fWebHistogram(0), b_fH(0),
            b_event_fTriggerBits(0), b_event_fIsValid(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TSelEvent::TSelEvent()
          : fReadType(0), fDebug(kFALSE), fCHist(0), fPtHist(0),
            fNTracksHist(0), fEventName(0), fTracks(0), fHighPt(0), fMuons(0),
            fH(0), b_event_fType(0), b_fEventName(0), b_event_fNtrack(0), b_event_fNseg(0),
            b_event_fNvertex(0), b_event_fFlag(0), b_event_fTemperature(0),
            b_event_fMeasures(0), b_event_fMatrix(0), b_fClosestDistance(0),
            b_event_fEvtHdr(0), b_fTracks(0), b_fHighPt(0), b_fMuons(0),
            b_event_fLastTrack(0), b_event_fWebHistogram(0), b_fH(0),
            b_event_fTriggerBits(0), b_event_fIsValid(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// The Begin() function is called at the start of the query.
/// When running with PROOF Begin() is only called on the client.
/// The tree argument is deprecated (on PROOF 0 is passed).

void TSelEvent::Begin(TTree *)
{
   TString option = GetOption();

   //get parameters

   Bool_t found_readtype=kFALSE;
   Bool_t found_debug=kFALSE;

   TIter nxt(fInput);
   TString sinput;
   TObject *obj;
   while ((obj = nxt())){
      sinput=obj->GetName();
      //Info("Begin", "name=%s", sinput.Data());
      if (sinput.Contains("PROOF_Benchmark_ReadType")){
         if ((fReadType = dynamic_cast<TPBReadType *>(obj))) found_readtype = kTRUE;
         continue;
      }
      if (sinput.Contains("PROOF_BenchmarkDebug")){
         TParameter<Int_t>* a=dynamic_cast<TParameter<Int_t>*>(obj);
         if (a){
            fDebug= a->GetVal();
            found_debug=kTRUE;
            //Info("Begin", "PROOF_BenchmarkDebug=%d", fDebug);
         }
         else{
            Error("Begin", "PROOF_BenchmarkDebug not type TParameter<Int_t>*");
         }
         continue;
      }
   }

   if (!found_readtype){
      fReadType = new TPBReadType(TPBReadType::kReadOpt);
      Warning("Begin", "PROOF_Benchmark_ReadType not found; using default: %d",
                       fReadType->GetType());
   }
   if (!found_debug){
      Warning("Begin", "PROOF_BenchmarkDebug not found; using default: %d",
                       fDebug);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// The SlaveBegin() function is called after the Begin() function.
/// When running with PROOF SlaveBegin() is called on each slave server.
/// The tree argument is deprecated (on PROOF 0 is passed).

void TSelEvent::SlaveBegin(TTree *tree)
{
   Init(tree);

   TString option = GetOption();

   Bool_t found_readtype=kFALSE;
   Bool_t found_debug=kFALSE;

   //fInput->Print("A");
   TIter nxt(fInput);
   TString sinput;
   TObject *obj;
   while ((obj = nxt())){
      sinput=obj->GetName();
      //Info("SlaveBegin", "name=%s", sinput.Data());
      if (sinput.Contains("PROOF_Benchmark_ReadType")){
         if ((fReadType = dynamic_cast<TPBReadType *>(obj))) found_readtype = kTRUE;
         continue;
      }
      if (sinput.Contains("PROOF_BenchmarkDebug")){
         TParameter<Int_t>* a=dynamic_cast<TParameter<Int_t>*>(obj);
         if (a){
            fDebug= a->GetVal();
            found_debug=kTRUE;
            //Info("SlaveBegin", "PROOF_BenchmarkDebug=%d", fDebug);
         }
         else{
            Error("SlaveBegin", "PROOF_BenchmarkDebug not type TParameter"
                                "<Int_t>*");
         }
         continue;
      }
   }

   if (!found_readtype){
      fReadType = new TPBReadType(TPBReadType::kReadOpt);
      Warning("SlaveBegin", "PROOF_Benchmark_ReadType not found; using default: %d",
                       fReadType->GetType());
   }
   if (!found_debug){
      Warning("SlaveBegin", "PROOF_BenchmarkDebug not found; using default: %d",
                            fDebug);
   }

   fPtHist = new TH1F("pt_dist","p_{T} Distribution", 100, 0, 5);
   fPtHist->SetDirectory(0);
   fPtHist->GetXaxis()->SetTitle("p_{T}");
   fPtHist->GetYaxis()->SetTitle("dN/p_{T}dp_{T}");

   fNTracksHist = new TH1F("ntracks_dist","N_{Tracks} per Event"
                           " Distribution", 100, 50, 150);
   //enable rebinning
   fNTracksHist->SetCanExtend(TH1::kAllAxes);
   fNTracksHist->SetDirectory(0);
   fNTracksHist->GetXaxis()->SetTitle("N_{Tracks}");
   fNTracksHist->GetYaxis()->SetTitle("N_{Events}");
}

////////////////////////////////////////////////////////////////////////////////
/// The Process() function is called for each entry in the tree (or possibly
/// keyed object in the case of PROOF) to be processed. The entry argument
/// specifies which entry in the currently loaded tree is to be processed.
/// It can be passed to either TTree::GetEntry() or TBranch::GetEntry()
/// to read either all or the required parts of the data. When processing
/// keyed objects with PROOF, the object is already loaded and is available
/// via the fObject pointer.
///
/// This function should contain the "body" of the analysis. It can contain
/// simple or elaborate selection criteria, run algorithms on the data
/// of the event and typically fill histograms.

Bool_t TSelEvent::Process(Long64_t entry)
{
   // WARNING when a selector is used with a TChain, you must use
   //  the pointer to the current TTree to call GetEntry(entry).
   //  The entry is always the local entry number in the current tree.
   //  Assuming that fChain is the pointer to the TChain being processed,
   //  use fChain->GetTree()->GetEntry(entry).

   if (fReadType->GetType() != TPBReadType::kReadNotSpecified){
      switch (fReadType->GetType()){
         case TPBReadType::kReadFull:
            // Full read
            fChain->GetTree()->GetEntry(entry);
            fNTracksHist->Fill(fNtrack);

            for(Int_t j=0;j<fTracks->GetEntries();j++){
               Track* curtrack = dynamic_cast<Track*>(fTracks->At(j));
               fPtHist->Fill(curtrack->GetPt(),1./curtrack->GetPt());
            }
            fTracks->Clear("C");
            break;
         case TPBReadType::kReadOpt:
            // Partial read
            b_event_fNtrack->GetEntry(entry);

            fNTracksHist->Fill(fNtrack);

            if (fNtrack>0) {
               b_fTracks->GetEntry(entry);
               for(Int_t j=0;j<fTracks->GetEntries();j++){
                  Track* curtrack = dynamic_cast<Track*>(fTracks->At(j));
                  fPtHist->Fill(curtrack->GetPt(),1./curtrack->GetPt());
               }
               fTracks->Clear("C");
            }
            break;
         case TPBReadType::kReadNo:
            // No read
            break;
         default:
            Error("Process", "Read type not supported; %d", fReadType->GetType());
            return kFALSE;
            break;
      }
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// The SlaveTerminate() function is called after all entries or objects
/// have been processed. When running with PROOF SlaveTerminate() is called
/// on each slave server.

void TSelEvent::SlaveTerminate()
{
}

////////////////////////////////////////////////////////////////////////////////
/// The Terminate() function is the last function to be called during
/// a query. It always runs on the client, it can be used to present
/// the results graphically or save the results to file.

void TSelEvent::Terminate()
{
}
