// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelEventGen                                                         //
//                                                                      //
// PROOF selector for event file generation.                            //
// List of files to be generated for each node is provided by client.   //
// And list of files generated is sent back.                            //
// Existing files are reused if not forced to be regenerated.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#define TSelEventGen_cxx

#include "TSelEventGen.h"
#include "TParameter.h"
#include "TProofNodeInfo.h"
#include "TProofBenchTypes.h"
#include "TProof.h"
#include "TMap.h" 
#include "TDSet.h"
#include "TEnv.h"
#include "TFileInfo.h"
#include "TFile.h"
#include "TSortedList.h"
#include "TRandom.h"
#include "Event.h"
#include "TProofServ.h"

ClassImp(TSelEventGen)

//______________________________________________________________________________
TSelEventGen::TSelEventGen()
             : fBaseDir(""), fNEvents(100000), fNTracks(100), fNTracksMax(-1),
               fRegenerate(kFALSE), fTotalGen(0), fFilesGenerated(0), fChain(0)
{
   // Constructor
   if (gProofServ){
      fBaseDir=gProofServ->GetDataDir();
      // Two directories up
      fBaseDir.Remove(fBaseDir.Last('/'));
      fBaseDir.Remove(fBaseDir.Last('/'));
   }
   else{
      fBaseDir="";
   }
}

//______________________________________________________________________________
void TSelEventGen::Begin(TTree *)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();
   // Determine the test type
   TMap *filemap = dynamic_cast<TMap *>
                     (fInput->FindObject("PROOF_FilesToProcess"));
   if (filemap) {
      //Info("Begin", "dumping the file map:");
      //filemap->Print();
   } else {
      if (fInput->FindObject("PROOF_FilesToProcess")) {
         Error("Begin", "object 'PROOF_FilesToProcess' found but not a map"
              " (%s)", fInput->FindObject("PROOF_FilesToProcess")->ClassName());
      } else {
         Error("Begin", "object 'PROOF_FilesToProcess' not found");
      }
   }
}

//______________________________________________________________________________
void TSelEventGen::SlaveBegin(TTree *tree)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   Init(tree);

   TString option = GetOption();

   //get parameters

   Bool_t found_basedir=kFALSE;
   Bool_t found_nevents=kFALSE;
   Bool_t found_ntracks=kFALSE;
   Bool_t found_ntrkmax=kFALSE;
   Bool_t found_regenerate=kFALSE;

   TIter nxt(fInput);
   TString sinput;
   TObject *obj;

   while ((obj = nxt())){

      sinput=obj->GetName();
      //Info("SlaveBegin", "Input list: %s", sinput.Data());

      if (sinput.Contains("PROOF_BenchmarkBaseDir")){
         TNamed* a=dynamic_cast<TNamed*>(obj);
         if (a){
            TString proof_benchmarkbasedir=a->GetTitle();
            if (!proof_benchmarkbasedir.IsNull()){
               TUrl u(proof_benchmarkbasedir, kTRUE);
               Bool_t isLocal = !strcmp(u.GetProtocol(), "file") ? kTRUE : kFALSE;
               if (isLocal && !gSystem->IsAbsoluteFileName(u.GetFile()))
                  u.SetFile(TString::Format("%s/%s", fBaseDir.Data(), u.GetFile())); 
               if ((gSystem->AccessPathName(u.GetFile()) &&
                    gSystem->mkdir(u.GetFile(), kTRUE) == 0) ||
                    gSystem->AccessPathName(u.GetFile(), kWritePermission)) {
                    // Directory is writable
                    fBaseDir = u.GetFile();
                    Info("SlaveBegin", "Using directory \"%s\"", fBaseDir.Data());
               } else{
                  // Directory is not writable or not available, use default directory
                  Warning("BeginSlave", "\"%s\" directory is not writable or not existing,"
                          " using default directory: %s",
                          proof_benchmarkbasedir.Data(), fBaseDir.Data());
               }
            } else{
               Info("BeginSlave", "Using default directory: %s",
                                   fBaseDir.Data());
            }
            found_basedir=kTRUE; 
         }
         else{
            Error("SlaveBegin", "PROOF_BenchmarkBaseDir not type TNamed");
         }
         continue;
      }
      if (sinput.Contains("PROOF_BenchmarkNEvents")){
         TParameter<Long64_t>* a=dynamic_cast<TParameter<Long64_t>*>(obj);
         if (a){
            fNEvents= a->GetVal();
            found_nevents=kTRUE; 
         }
         else{
            Error("SlaveBegin", "PROOF_BenchmarkEvents not type TParameter"
                                "<Long64_t>*");
         }
         continue;
      }
      if (sinput.Contains("PROOF_BenchmarkNTracks")){
         TParameter<Int_t>* a=dynamic_cast<TParameter<Int_t>*>(obj);
         if (a){
            fNTracks=a->GetVal();
            found_ntracks=kTRUE; 
         }
         else{
            Error("SlaveBegin", "PROOF_BenchmarkNTracks not type TParameter"
                                "<Int_t>*");
         }
         continue;
      }
      if (sinput.Contains("PROOF_BenchmarkNTracksMax")){
         TParameter<Int_t>* a=dynamic_cast<TParameter<Int_t>*>(obj);
         if (a){
            fNTracksMax=a->GetVal();
            found_ntrkmax=kTRUE; 
         }
         else{
            Error("SlaveBegin", "PROOF_BenchmarkNTracksMax not type TParameter"
                                "<Int_t>*");
         }
         continue;
      }
      if (sinput.Contains("PROOF_BenchmarkRegenerate")){
         TParameter<Int_t>* a=dynamic_cast<TParameter<Int_t>*>(obj);
         if (a){
            fRegenerate=a->GetVal();
            found_regenerate=kTRUE; 
         }
         else{
            Error("SlaveBegin", "PROOF_BenchmarkRegenerate not type TParameter"
                                "<Int_t>*");
         }
         continue;
      }
   }
   
   if (!found_basedir){
      Warning("SlaveBegin", "PROOF_BenchmarkBaseDir not found; using default:"
                            " %s", fBaseDir.Data());
   }
   if (!found_nevents){
      Warning("SlaveBegin", "PROOF_BenchmarkNEvents not found; using default:"
                            " %lld", fNEvents);
   }
   if (!found_ntracks){
      Warning("SlaveBegin", "PROOF_BenchmarkNTracks not found; using default:"
                            " %d", fNTracks);
   }
   if (!found_ntrkmax){
      Warning("SlaveBegin", "PROOF_BenchmarkNTracksMax not found; using default:"
                            " %d", fNTracksMax);
   } else if (fNTracksMax <= fNTracks) {
      Warning("SlaveBegin", "PROOF_BenchmarkNTracksMax must be larger then"
                            " fNTracks=%d ; ignoring", fNTracks);
      fNTracksMax = -1;
      found_ntrkmax = kFALSE;
   }
   if (!found_regenerate){
      Warning("SlaveBegin", "PROOF_BenchmarkRegenerate not found; using"
                            " default: %d", fRegenerate);
   }

   fFilesGenerated = new TList();

   TString hostname(TUrl(gSystem->HostName()).GetHostFQDN());
   TString thisordinal = gProofServ ? gProofServ->GetOrdinal() : "n.d";
   TString sfilegenerated =
      TString::Format("PROOF_FilesGenerated_%s_%s", hostname.Data(), thisordinal.Data());
   fFilesGenerated->SetName(sfilegenerated);
   fFilesGenerated->SetOwner(kTRUE);
}

//______________________________________________________________________________
Long64_t TSelEventGen::GenerateFiles(TString filename, Long64_t sizenevents)
{
//Generate files for IO-bound run
//Input parameters
//   filename: The name of the file to be generated
//   sizenevents: Either the number of events to generate when
//                filetype==kPBFileBenchmark
//                or the size of the file to generate when
//                filetype==kPBFileCleanup
//Returns
//   Either Number of entries in the file when
//   filetype==kPBFileBenchmark
//   or bytes written when filetype==kPBFileCleanup
//return 0 in case error

   Info("GenerateFiles","file: %s", filename.Data());
    
   Long64_t nentries=0;
   TDirectory* savedir = gDirectory;
   //printf("current dir=%s\n", gDirectory->GetPath());

   TFile *f = TFile::Open(filename, "RECREATE");

   savedir->cd();

   if (!f || f->IsZombie()) return 0;

   Event *event=new Event();
   Event *ep = event;
   TTree* eventtree= new TTree("EventTree", "Event Tree");
   eventtree->SetDirectory(f);

   const Int_t buffersize=32000;
   eventtree->Branch("event", "Event", &ep, buffersize, 1);
   eventtree->AutoSave();

   Long64_t i=0;
   Long64_t size_generated=0;

//   f->SetCompressionLevel(0); //no compression
   Int_t ntrks = fNTracks;
   
   Info("GenerateFiles", "Generating %s", filename.Data());   
   while (sizenevents--){
      //event->Build(i++,fNTracksBench,0);
      if (fNTracksMax > fNTracks) {
         // Required to smear the number of tracks between [min,max]
         ntrks = fNTracks + gRandom->Integer(fNTracksMax - fNTracks);
      }
      event->Build(i++, ntrks, 0);
      size_generated+=eventtree->Fill();
   }
   nentries=eventtree->GetEntries();
   Info("GenerateFiles", "%s generated with %lld entries", filename.Data(),
                                                              nentries);
   savedir = gDirectory;

   f = eventtree->GetCurrentFile();
   f->cd();
   eventtree->Write();
   eventtree->SetDirectory(0);

   f->Close();
   delete f;
   f = 0;
   eventtree->Delete();
   event->Delete();
   savedir->cd();

   return nentries;
}

//______________________________________________________________________________
Bool_t TSelEventGen::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either TTree::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.

   // WARNING when a selector is used with a TChain, you must use
   //  the pointer to the current TTree to call GetEntry(entry).
   //  The entry is always the local entry number in the current tree.
   //  Assuming that fChain is the pointer to the TChain being processed,
   //  use fChain->GetTree()->GetEntry(entry).

   TDSetElement *fCurrent = 0;
   TPair *elemPair = 0;
   if (fInput && (elemPair = dynamic_cast<TPair *>
                      (fInput->FindObject("PROOF_CurrentElement")))) {
      if ((fCurrent = dynamic_cast<TDSetElement *>(elemPair->Value()))) {
         Info("Process", "entry %lld: file: '%s'", entry, fCurrent->GetName());
      } else {
         Error("Process", "entry %lld: no file specified!", entry);
         return kFALSE;
      }
   }

   // Generate
   TString filename =
      TString::Format("%s/%s", fBaseDir.Data(), fCurrent->GetName());
      
   // Set the Url for remote access
   TString dsrv, seed;
   seed = TString::Format("%s/%s", gSystem->HostName(), filename.Data());
   if (gSystem->Getenv("LOCALDATASERVER")) {
      dsrv = gSystem->Getenv("LOCALDATASERVER");
      if (!dsrv.EndsWith("/")) dsrv += "/";
   } else {
      dsrv.Form("root://%s/", TUrl(gSystem->HostName()).GetHostFQDN());
   }
   TString srvProto = TUrl(dsrv).GetProtocol();
      
   // Remove prefix, if any, if included and if Xrootd
   TString fndset(filename);
   TString pfx  = gEnv->GetValue("Path.Localroot","");
   if (!pfx.IsNull() && fndset.BeginsWith(pfx) &&
      (srvProto == "root" || srvProto == "xrd")) fndset.Remove(0, pfx.Length());
   
   TFileInfo *fi = new TFileInfo(TString::Format("%s%s", dsrv.Data(), fndset.Data()));

   //generate files
   Long64_t neventstogenerate = fNEvents;

   Bool_t filefound=kFALSE;
   FileStat_t filestat;
   if (!fRegenerate && !gSystem->GetPathInfo(filename, filestat)) { //stat'ed
      TFile *f = TFile::Open(filename);
      if (f && !f->IsZombie()){
         TTree* t = (TTree *) f->Get("EventTree");
         if (t) {
            Long64_t entries_file=t->GetEntries();
            if (entries_file == neventstogenerate) {
               //file size seems to be correct, skip generation
               Info("Process", "Bench file (%s, entries=%lld) exists:"
                               " skipping generation.", fi->GetFirstUrl()->GetFile(),
                               entries_file);
               neventstogenerate -= entries_file;
               if (fFilesGenerated) {
                  // Set file size and mark it staged 
                  fi->SetSize(f->GetSize());
                  fi->SetBit(TFileInfo::kStaged);
                  // Add meta data to the file
                  TFileInfoMeta* fimeta = new TFileInfoMeta("/EventTree", "TTree", entries_file);
                  fi->AddMetaData(fimeta);
                  // Add the fileinfo to the list
                  fFilesGenerated->Add(fi);
                  filefound = kTRUE;
               }
            }
         }
         f->Close();
      }
      SafeDelete(f);
   }

   if (!filefound) {
      gRandom->SetSeed(static_cast<UInt_t>(TMath::Hash(seed)));
      Long64_t entries_file = GenerateFiles(filename, neventstogenerate);
      neventstogenerate -= entries_file;

      TFile *f = TFile::Open(filename);
      if (f && !f->IsZombie()) {
         // Set file size and mark it staged 
         fi->SetSize(f->GetSize());
         fi->SetBit(TFileInfo::kStaged);
         f->Close();
         // Add meta data to the file
         TFileInfoMeta* fimeta = new TFileInfoMeta("/EventTree", "TTree", entries_file);
         fi->AddMetaData(fimeta);
         if (fFilesGenerated) fFilesGenerated->Add(fi);
      }
      SafeDelete(f);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TSelEventGen::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server
   if (fFilesGenerated && fFilesGenerated->GetSize() > 0) {
      fOutput->Add(fFilesGenerated);
      Warning("SlaveTerminate",
              "list '%s' of files generated by this worker added to the output list",
              fFilesGenerated->GetName());
   } else {
      if (!fFilesGenerated) {
         Warning("SlaveTerminate", "no list of generated files defined!");
      } else {
         Warning("SlaveTerminate", "list of generated files is empty!");
      }
   }
}

//______________________________________________________________________________
void TSelEventGen::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.
}

//______________________________________________________________________________
void TSelEventGen::Print(Option_t *) const
{

   Printf("fNEvents=%lld", fNEvents);
   Printf("fBaseDir=%s", fBaseDir.Data());
   Printf("fNTracks=%d", fNTracks);
   Printf("fRegenerate=%d", fRegenerate);
}

