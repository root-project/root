#define ProofAux_cxx

//////////////////////////////////////////////////////////////
//
// Selector used for auxilliary actions in the PROOF tutorials
//
//////////////////////////////////////////////////////////////

#include "ProofAux.h"
#include "TDSet.h"
#include "TProofServ.h"
#include "TMap.h"
#include "TString.h"
#include "TSystem.h"
#include "TParameter.h"
#include "TFile.h"
#include "TUrl.h"
#include "TTree.h"
#include "TRandom.h"
#include "TMath.h"

//_____________________________________________________________________________
ProofAux::ProofAux()
{
   // Constructor

   fAction = -1;
   fNEvents= -1;
   fMainList = 0;
   fFriendList = 0;
}

//_____________________________________________________________________________
ProofAux::~ProofAux()
{
   // Destructor

}

//_____________________________________________________________________________
Int_t ProofAux::GetAction(TList *input)
{
   // Get the required action.
   // Returns -1 if unknown.

   Int_t action = -1;
   // Determine the test type
   TNamed *ntype = dynamic_cast<TNamed*>(input->FindObject("ProofAux_Action"));
   if (ntype) {
      if (!strcmp(ntype->GetTitle(), "GenerateTrees")) {
         action = 0;
      } else if (!strcmp(ntype->GetTitle(), "GenerateTreesSameFile")) {
         action = 1;
      } else {
         Warning("GetAction", "unknown action: '%s'", ntype->GetTitle());
      }
   }
   // Done
   return action;
}


//_____________________________________________________________________________
void ProofAux::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Determine the action type
   fAction = GetAction(fInput);
}

//_____________________________________________________________________________
void ProofAux::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // Determine the action type
   fAction = GetAction(fInput);

   // Get the number of events
   TParameter<Long64_t> *a = (TParameter<Long64_t> *) fInput->FindObject("ProofAux_NEvents");
   if (a) fNEvents = a->GetVal();

   // Create lists
   fMainList = new TList;
   if (gProofServ) fMainList->SetName(TString::Format("MainList-%s", gProofServ->GetOrdinal()));
   fFriendList = new TList;
   if (gProofServ) fFriendList->SetName(TString::Format("FriendList-%s", gProofServ->GetOrdinal()));
}

//_____________________________________________________________________________
Bool_t ProofAux::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofAux::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.
   //
   // The processing can be stopped by calling Abort().
   //
   // Use fStatus to set the return value of TTree::Process().
   //
   // The return value is currently not used.

   // Nothing to do if the action if not defined
   if (fAction < 0) {
      Error("Process", "action not specified!");
      return kFALSE;
   }

   // Link to current element, if any
   TDSetElement *fCurrent = 0;
   TPair *elemPair = 0;
   if (fInput && (elemPair = dynamic_cast<TPair *>(fInput->FindObject("PROOF_CurrentElement")))) {
      if ((fCurrent = dynamic_cast<TDSetElement *>(elemPair->Value()))) {
         Info("Process", "entry %lld: file: '%s'", entry, fCurrent->GetName());
      } else {
         Error("Process", "entry %lld: no file specified!", entry);
         return kFALSE;
      }
   }

   // Act now
   if (fAction == 0) {
      TString fnt;
      // Generate the TTree and save it in the specified file
      if (GenerateTree(fCurrent->GetName(), fNEvents, fnt) != 0) {
         Error("Process", "problems generating tree (%lld, %s, %lld)",
                          entry, fCurrent->GetName(), fNEvents);
         return kFALSE;
      }
      // The output filename
      TString fnf(fnt);
      TString xf = gSystem->BaseName(fnf);
      fnf = gSystem->DirName(fnf);
      if (xf.Contains("tree")) {
         xf.ReplaceAll("tree", "friend");
      } else {
         if (xf.EndsWith(".root")) {
            xf.ReplaceAll(".root", "_friend.root");
         } else {
            xf += "_friend";
         }
      }
      fnf += TString::Format("/%s", xf.Data());
      // Generate the TTree friend and save it in the specified file
      if (GenerateFriend(fnt, fnf) != 0) {
         Error("Process", "problems generating friend tree for %s (%s)",
                          fCurrent->GetName(), fnt.Data());
         return kFALSE;
      }
   } else if (fAction == 1) {
      TString fnt;
      // Generate the TTree and save it in the specified file
      if (GenerateTree(fCurrent->GetName(), fNEvents, fnt) != 0) {
         Error("Process", "problems generating tree (%lld, %s, %lld)",
                          entry, fCurrent->GetName(), fNEvents);
         return kFALSE;
      }
      // Generate the TTree friend and save it in the specified file
      if (GenerateFriend(fnt) != 0) {
         Error("Process", "problems generating friend tree for %s (%s)",
                          fCurrent->GetName(), fnt.Data());
         return kFALSE;
      }
   } else {
      // Unknown action
      Warning("Process", "do not know how to process action %d - do nothing", fAction);
      return kFALSE;
   }

   return kTRUE;
}

//_____________________________________________________________________________
void ProofAux::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

   if (fMainList && fMainList->GetSize() > 0) fOutput->Add(fMainList);
   if (fFriendList && fFriendList->GetSize() > 0) fOutput->Add(fFriendList);
}

//_____________________________________________________________________________
void ProofAux::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

}

//_____________________________________________________________________________
Int_t ProofAux::GenerateTree(const char *fnt, Long64_t ent, TString &fn)
{
   // Generate the main tree for the 'friends' tutorial; the tree is called
   // 'Tmain', has 'ent' entries and is saved to file 'fnt'.
   // The full file path is returned in 'fn'.
   // Return 0 on success, -1 on error.

   Int_t rc = -1;

   // Check the filename
   fn = fnt;
   if (fn.IsNull()) {
      Error("GenerateTree", "file name undefined!");
      return rc;
   }
   TUrl uu(fn, kTRUE);
   if (!strcmp(uu.GetProtocol(), "file") && !fn.BeginsWith("/")) {
      // Local file with relative path: create under the data directory
      if (!gProofServ ||
          !(gProofServ->GetDataDir()) || strlen(gProofServ->GetDataDir()) <= 0) {
         Error("GenerateTree", "data directory undefined!");
         return rc;
      }
      // Insert data directory
      fn.Insert(0, TString::Format("%s/", gProofServ->GetDataDir()));
      // Make sure the directory exists
      TString dir = gSystem->DirName(fn);
      if (gSystem->AccessPathName(dir, kWritePermission)) {
         if (gSystem->mkdir(dir, kTRUE) != 0) {
            Error("GenerateTree", "problems creating directory %s to store the file", dir.Data());
            return rc;
         }
      }
   }

   // Create the file
   TDirectory* savedir = gDirectory;
   TFile *f = new TFile(fn, "RECREATE");
   if (!f || f->IsZombie()) {
      Error("GenerateTree", "problems opening file %s", fn.Data());
      return rc;
   }
   savedir->cd();
   rc = 0;

   // Create the tree
   TTree *T = new TTree("Tmain","Main tree for tutorial friends");
   T->SetDirectory(f);
   Int_t Run = 1;
   T->Branch("Run",&Run,"Run/I");
   Long64_t Event = 0;
   T->Branch("Event",&Event,"Event/L");
   Float_t x = 0., y = 0., z = 0.;
   T->Branch("x",&x,"x/F");
   T->Branch("y",&y,"y/F");
   T->Branch("z",&z,"z/F");
   TRandom r;
   for (Long64_t i = 0; i < ent; i++) {
      if (i > 0 && i%1000 == 0) Run++;
      Event = i;
      x = r.Gaus(10,1);
      y = r.Gaus(20,2);
      z = r.Landau(2,1);
      T->Fill();
   }
   T->Print();
   f->cd();
   T->Write();
   T->SetDirectory(0);
   f->Close();
   delete f;
   delete T;

   // Notify success
   Info("GenerateTree", "file '%s' successfully created", fn.Data());

   // Add to the list
   TString fds(fn);
   if (!strcmp(uu.GetProtocol(), "file"))
      fds.Insert(0, TString::Format("root://%s/", gSystem->HostName()));
   fMainList->Add(new TObjString(fds));

   // Done
   return rc;
}

//_____________________________________________________________________________
Int_t ProofAux::GenerateFriend(const char *fnt, const char *fnf)
{
   // Generate the friend tree for the main tree in the 'friends' tutorial fetched
   // from 'fnt'.
   // the tree is called 'Tfriend', has the same number of entries as the main
   // tree and is saved to file 'fnf'. If 'fnf' is not defined the filename is
   // derived from 'fnt' either replacing 'tree' with 'friend', or adding '_friend'
   // before the '.root' extension.
   // Return 0 on success, -1 on error.

   Int_t rc = -1;
   // Check the input filename
   TString fin(fnt);
   if (fin.IsNull()) {
      Error("GenerateFriend", "file name for the main tree undefined!");
      return rc;
   }
   // Make sure that the file can be read
   if (gSystem->AccessPathName(fin, kReadPermission)) {
      Error("GenerateFriend", "input file does not exist or cannot be read: %s", fin.Data());
      return rc;
   }

   // File handlers
   Bool_t sameFile = kTRUE;
   const char *openMain = "UPDATE";

   // The output filename
   TString fout(fnf);
   if (!fout.IsNull()) {
      sameFile = kFALSE;
      openMain = "READ";
      // Make sure the directory exists
      TString dir = gSystem->DirName(fout);
      if (gSystem->AccessPathName(dir, kWritePermission)) {
         if (gSystem->mkdir(dir, kTRUE) != 0) {
            Error("GenerateFriend", "problems creating directory %s to store the file", dir.Data());
            return rc;
         }
      }
   } else {
      // We set the same name
      fout = fin;
   }

   // Get main tree
   TFile *fi = TFile::Open(fin, openMain);
   if (!fi || fi->IsZombie()) {
      Error("GenerateFriend", "problems opening input file %s", fin.Data());
      return rc;
   }
   TTree *Tin = (TTree *) fi->Get("Tmain");
   if (!Tin) {
      Error("GenerateFriend", "problems getting tree 'Tmain' from file %s", fin.Data());
      delete fi;
      return rc;
   }
   // Set branches
   Float_t x, y, z;
   Tin->SetBranchAddress("x", &x);
   Tin->SetBranchAddress("y", &y);
   Tin->SetBranchAddress("z", &z);
   TBranch *b_x = Tin->GetBranch("x");
   TBranch *b_y = Tin->GetBranch("y");
   TBranch *b_z = Tin->GetBranch("z");

   TDirectory* savedir = gDirectory;
   // Create output file
   TFile *fo = 0;
   if (!sameFile) {
      fo = new TFile(fout, "RECREATE");
      if (!fo || fo->IsZombie()) {
         Error("GenerateFriend", "problems opening file %s", fout.Data());
         delete fi;
         return rc;
      }
      savedir->cd();
   } else {
      // Same file
      fo = fi;
   }
   rc = 0;

   // Create the tree
   TTree *Tfrnd = new TTree("Tfrnd", "Friend tree for tutorial 'friends'");
   Tfrnd->SetDirectory(fo);
   Float_t r = 0;
   Tfrnd->Branch("r",&x,"r/F");
   Long64_t ent = Tin->GetEntries();
   for (Long64_t i = 0; i < ent; i++) {
      b_x->GetEntry(i);
      b_y->GetEntry(i);
      b_z->GetEntry(i);
      r = TMath::Sqrt(x*x + y*y + z*z);
      Tfrnd->Fill();
   }
   if (!sameFile) {
      fi->Close();
      delete fi;
   }
   Tfrnd->Print();
   fo->cd();
   Tfrnd->Write();
   Tfrnd->SetDirectory(0);
   fo->Close();
   delete fo;
   delete Tfrnd;

   // Notify success
   Info("GenerateFriend", "friend file '%s' successfully created", fout.Data());

   // Add to the list
   TUrl uu(fout);
   if (!strcmp(uu.GetProtocol(), "file"))
      fout.Insert(0, TString::Format("root://%s/", gSystem->HostName()));
   fFriendList->Add(new TObjString(fout));

   // Done
   return rc;
}
