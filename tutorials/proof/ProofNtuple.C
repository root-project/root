/// \file
/// \ingroup tutorial_proofntuple
///
/// Selector to fill a simple ntuple
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)

#define ProofNtuple_cxx

#include "ProofNtuple.h"
#include <TCanvas.h>
#include <TFrame.h>
#include <TPaveText.h>
#include <TMath.h>
#include <TNtuple.h>
#include <TRandom3.h>
#include <TROOT.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TFile.h>
#include <TProofOutputFile.h>

//_____________________________________________________________________________
ProofNtuple::~ProofNtuple()
{
   // Destructor

   SafeDelete(fNtp);
   SafeDelete(fNtp2);
   SafeDelete(fFile);
   SafeDelete(fRandom);
}

//_____________________________________________________________________________
void ProofNtuple::PlotNtuple(TNtuple *ntp, const char *ntptitle)
{
   // Make some plots from the ntuple 'ntp'

   //
   // Create a canvas, with 2 pads
   //
   TCanvas *c1 = new TCanvas(Form("cv-%s", ntp->GetName()), ntptitle,800,10,700,780);
   c1->Divide(1,2);
   TPad *pad1 = (TPad *) c1->GetPad(1);
   TPad *pad2 = (TPad *) c1->GetPad(2);
   //
   // Display a function of one ntuple column imposing a condition
   // on another column.
   pad1->cd();
   pad1->SetGrid();
   pad1->SetLogy();
   pad1->GetFrame()->SetFillColor(15);
   ntp->SetLineColor(1);
   ntp->SetFillStyle(1001);
   ntp->SetFillColor(45);
   ntp->Draw("3*px+2","px**2+py**2>1");
   ntp->SetFillColor(38);
   ntp->Draw("2*px+2","pz>2","same");
   ntp->SetFillColor(5);
   ntp->Draw("1.3*px+2","(px^2+py^2>4) && py>0","same");
   pad1->RedrawAxis();

   //
   // Display a 3-D scatter plot of 3 columns. Superimpose a different selection.
   pad2->cd();
   ntp->Draw("pz:py:px","(pz<10 && pz>6)+(pz<4 && pz>3)");
   ntp->SetMarkerColor(4);
   ntp->Draw("pz:py:px","pz<6 && pz>4","same");
   ntp->SetMarkerColor(5);
   ntp->Draw("pz:py:px","pz<4 && pz>3","same");
   TPaveText *l2 = new TPaveText(0.,0.6,0.9,0.95);
   l2->SetFillColor(42);
   l2->SetTextAlign(12);
   l2->AddText("You can interactively rotate this view in 2 ways:");
   l2->AddText("  - With the RotateCube in clicking in this pad");
   l2->AddText("  - Selecting View with x3d in the View menu");
   l2->Draw();

   // Final update
   c1->cd();
   c1->Update();
}

//_____________________________________________________________________________
void ProofNtuple::Begin(TTree * /*tree*/)
{
   // The Begin() function is called at the start of the query.
   // When running with PROOF Begin() is only called on the client.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   TNamed *out = (TNamed *) fInput->FindObject("PROOF_NTUPLE_DONT_PLOT");
   if (out) fPlotNtuple = kFALSE;
}

//_____________________________________________________________________________
void ProofNtuple::SlaveBegin(TTree * /*tree*/)
{
   // The SlaveBegin() function is called after the Begin() function.
   // When running with PROOF SlaveBegin() is called on each slave server.
   // The tree argument is deprecated (on PROOF 0 is passed).

   TString option = GetOption();

   // We may be creating a dataset or a merge file: check it
   TNamed *nm = dynamic_cast<TNamed *>(fInput->FindObject("SimpleNtuple.root"));
   if (nm) {
      // Just create the object
      UInt_t opt = TProofOutputFile::kRegister | TProofOutputFile::kOverwrite | TProofOutputFile::kVerify;
      fProofFile = new TProofOutputFile("SimpleNtuple.root",
                                        TProofOutputFile::kDataset, opt, nm->GetTitle());
   } else {
      // For the ntuple, we use the automatic file merging facility
      // Check if an output URL has been given
      TNamed *out = (TNamed *) fInput->FindObject("PROOF_OUTPUTFILE_LOCATION");
      Info("SlaveBegin", "PROOF_OUTPUTFILE_LOCATION: %s", (out ? out->GetTitle() : "undef"));
      fProofFile = new TProofOutputFile("SimpleNtuple.root", (out ? out->GetTitle() : "M"));
      out = (TNamed *) fInput->FindObject("PROOF_OUTPUTFILE");
      if (out) fProofFile->SetOutputFileName(out->GetTitle());
   }

   // Open the file
   fFile = fProofFile->OpenFile("RECREATE");
   if (fFile && fFile->IsZombie()) SafeDelete(fFile);

   // Cannot continue
   if (!fFile) {
      Info("SlaveBegin", "could not create '%s': instance is invalid!", fProofFile->GetName());
      return;
   }

   // Now we create the ntuple
   fNtp = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");
   // File resident
   fNtp->SetDirectory(fFile);
   fNtp->AutoSave();

   // Now we create the second ntuple
   fNtp2 = new TNtuple("ntuple2","Demo ntuple2","vx:vy:vz");
   // File resident
   fNtp2->SetDirectory(fFile);
   fNtp2->AutoSave();

   // Should we generate the random numbers or take them from the ntuple ?
   TNamed *unr = (TNamed *) fInput->FindObject("PROOF_USE_NTP_RNDM");
   if (unr) {
      // Get the ntuple from the input list
      if (!(fNtpRndm = dynamic_cast<TNtuple *>(fInput->FindObject("NtpRndm")))) {
         Warning("SlaveBegin",
                 "asked to use rndm ntuple but 'NtpRndm' not found in the"
                 " input list! Using the random generator");
         fInput->Print();
      } else {
         Info("SlaveBegin", "taking randoms from input ntuple 'NtpRndm'");
      }
   }

   // Init the random generator, if required
   if (!fNtpRndm) fRandom = new TRandom3(0);
}

//_____________________________________________________________________________
Bool_t ProofNtuple::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either ProofNtuple::GetEntry() or TBranch::GetEntry()
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

   if (!fNtp) return kTRUE;

   // Fill ntuple
   Float_t px, py, random;
   if (fNtpRndm) {
      // Get the entry
      Float_t *ar = fNtpRndm->GetArgs();
      Long64_t ent = entry % fNtpRndm->GetEntries();
      fNtpRndm->GetEntry(ent);
      random = ar[0];
      px = (Float_t) TMath::ErfInverse((Double_t)(ar[1]*2 - 1.)) * TMath::Sqrt(2.);
      py = (Float_t) TMath::ErfInverse((Double_t)(ar[2]*2 - 1.)) * TMath::Sqrt(2.);
   } else if (fRandom) {
      fRandom->Rannor(px,py);
      random = fRandom->Rndm();
   } else {
      Abort("no way to get random numbers! Stop processing", kAbortProcess);
      return kTRUE;
   }
   Float_t pz = px*px + py*py;
   Int_t i = (Int_t) entry;
   fNtp->Fill(px,py,pz,random,i);

   if (!fNtp2) return kTRUE;

   // The second ntuple
   Float_t vz = random * 2. - 1.;
   fNtp2->Fill(px,py,vz);

   return kTRUE;
}

//_____________________________________________________________________________
void ProofNtuple::SlaveTerminate()
{
   // The SlaveTerminate() function is called after all entries or objects
   // have been processed. When running with PROOF SlaveTerminate() is called
   // on each slave server.

   // Write the ntuple to the file
   if (fFile) {
      if (!fNtp) {
         Error("SlaveTerminate", "'ntuple' is undefined!");
         return;
      }
      Bool_t cleanup = kFALSE;
      TDirectory *savedir = gDirectory;
      if (fNtp->GetEntries() > 0) {
         fFile->cd();
         fNtp->Write(0, TObject::kOverwrite);
         if (fNtp2 && fNtp2->GetEntries() > 0) fNtp2->Write(0, TObject::kOverwrite);
         fProofFile->Print();
         fOutput->Add(fProofFile);
      } else {
         cleanup = kTRUE;
      }
      fNtp->SetDirectory(0);
      if (fNtp2) fNtp2->SetDirectory(0);
      gDirectory = savedir;
      fFile->Close();
      // Cleanup, if needed
      if (cleanup) {
         TUrl uf(*(fFile->GetEndpointUrl()));
         SafeDelete(fFile);
         gSystem->Unlink(uf.GetFile());
         SafeDelete(fProofFile);
      }
   }
}

//_____________________________________________________________________________
void ProofNtuple::Terminate()
{
   // The Terminate() function is the last function to be called during
   // a query. It always runs on the client, it can be used to present
   // the results graphically or save the results to file.

   // Do nothing is not requested (dataset creation run)
   if (!fPlotNtuple) return;

   // Get the ntuple from the file
   if ((fProofFile =
           dynamic_cast<TProofOutputFile*>(fOutput->FindObject("SimpleNtuple.root")))) {

      TString outputFile(fProofFile->GetOutputFileName());
      TString outputName(fProofFile->GetName());
      outputName += ".root";
      Printf("outputFile: %s", outputFile.Data());

      // Read the ntuple from the file
      fFile = TFile::Open(outputFile);
      if (fFile) {
         Printf("Managed to open file: %s", outputFile.Data());
         fNtp = (TNtuple *) fFile->Get("ntuple");
      } else {
         Error("Terminate", "could not open file: %s", outputFile.Data());
      }
      if (!fFile) return;

   } else {
      Error("Terminate", "TProofOutputFile not found");
      return;
   }

   // Plot ntuples
   if (fNtp) PlotNtuple(fNtp, "proof ntuple");

}
