// @(#)root/proof:$Name:  $:$Id: TProofDraw.cxx,v 1.5 2004/12/28 20:51:25 brun Exp $
// Author: Maarten Ballintijn   24/09/2003

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofDraw                                                           //
//                                                                      //
// Implement Tree drawing using PROOF.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TProofDraw.h"


#include "TError.h"
#include "TH1F.h"
#include "TProofDebug.h"
#include "TStatus.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TTree.h"


ClassImp(TProofDraw)


//______________________________________________________________________________
TProofDraw::TProofDraw()
   : fStatus(0), fManager(0), fSelFormula(0), fVarXFormula(0), fHistogram(0)
{
}


//______________________________________________________________________________
TProofDraw::~TProofDraw()
{
   ClearFormulas();
   // delete fHistogram;
}


//______________________________________________________________________________
void TProofDraw::ClearFormulas()
{
   fManager = 0; // we forget about it, refcounting will do cleanup
   delete fSelFormula; fSelFormula = 0;
   delete fVarXFormula; fVarXFormula = 0;
}


//______________________________________________________________________________
void TProofDraw::Init(TTree *tree)
{
   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);

   ClearFormulas();

   fManager = new TTreeFormulaManager();

   fSelFormula = new TTreeFormula("Selection", fSelection, tree);
   fSelFormula->SetQuickLoad(kTRUE);
   if (fSelFormula->GetNdim() == 0) {
      SetError("Init", Form("selection invalid (%s)", fSelection.Data()));
      ClearFormulas();
      return;
   }

   if (fSelFormula->IsString()) {
      SetError("Init", Form("strings not supported, selection invalid (%s)", fSelection.Data()));
      ClearFormulas();
      return;
   }

   if (fSelFormula->EvalClass() != 0) {
      SetError("Init", Form("Objects not supported, selection invalid (%s)", fSelection.Data()));
      ClearFormulas();
      return;
   }

   fManager->Add(fSelFormula);

   PDB(kDraw,1) fSelFormula->Print();

   fVarXFormula = new TTreeFormula("VarX", fVarX, tree);
   fVarXFormula->SetQuickLoad(kTRUE);
   if (fVarXFormula->GetNdim() == 0) {
      SetError("Init", Form("varX invalid (%s)", fVarX.Data()));
      ClearFormulas();
      return;
   }

   if (fVarXFormula->IsString()) {
      SetError("Init", Form("strings not supported, varX invalid (%s)", fVarX.Data()));
      ClearFormulas();
      return;
   }

   if (fVarXFormula->EvalClass() != 0) {
      SetError("Init", Form("Objects not supported, varX invalid (%s)", fVarX.Data()));
      ClearFormulas();
      return;
   }

   fManager->Add(fVarXFormula);

   PDB(kDraw,1) fVarXFormula->Print();

   fManager->Sync();

   fTree = tree;
}


//______________________________________________________________________________
Bool_t TProofDraw::Notify()
{
   PDB(kDraw,1) Info("Notify","Enter");
   if (fStatus == 0) {
      fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"));
      Assert(fStatus);
   }

   if (!fStatus->IsOk()) return kFALSE;

   if (fVarXFormula) fVarXFormula->UpdateFormulaLeaves();
   if (fSelFormula) fSelFormula->UpdateFormulaLeaves();
   return kTRUE;
}


//______________________________________________________________________________
void TProofDraw::Begin(TTree *tree)
{
   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fVarX = fInput->FindObject("varexp")->GetTitle();

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fVarX.Data());

}


//______________________________________________________________________________
void TProofDraw::SlaveBegin(TTree *tree)
{
   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fVarX = fInput->FindObject("varexp")->GetTitle();

   fHistogram = new TH1F("htemp", Form("%s {%s}", fVarX.Data(), fSelection.Data()),
                          100, 0., 0.);
   fHistogram->SetDirectory(0);   // take ownership
   fOutput->Add(fHistogram);      // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fVarX.Data());
}


//______________________________________________________________________________
Bool_t TProofDraw::Process(Long64_t entry)
{
   PDB(kDraw,3) Info("Process","Enter entry = %d", entry);

   fTree->LoadTree(entry);
   Int_t ndata = fManager->GetNdata();

   PDB(kDraw,3) Info("Process","ndata = %d", ndata);

   for (Int_t i=0;i<ndata;i++) {
      Double_t w = fSelFormula->EvalInstance(i);

      PDB(kDraw,3) Info("Process","w[%d] = %f", i, w);

      if (w == 0.0) continue;
      Double_t x = fVarXFormula->EvalInstance(i);

      PDB(kDraw,3) Info("Process","x[%d] = %f", i, x);
      fHistogram->Fill(x, w);
   }

   return kTRUE;
}


//______________________________________________________________________________
void TProofDraw::SlaveTerminate(void)
{
   PDB(kDraw,1) Info("SlaveTerminate","Enter");

}


//______________________________________________________________________________
void TProofDraw::Terminate(void)
{
   PDB(kDraw,1) Info("Terminate","Enter");
   if (fStatus == 0) {
      fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"));
      if (fStatus == 0) {
         // did not run selector, error messages were already printed
         return;
      }
   }

   if (!fStatus->IsOk()) {
      fStatus->Print();
      return;
   }

   fHistogram = (TH1F *) fOutput->FindObject("htemp");
   if (fHistogram == 0) {
      Error("Terminate","Did not find histogram?");
      return;
   }

   fHistogram->Draw();
}


//______________________________________________________________________________
void TProofDraw::SetError(const char *sub, const char *mesg)
{
   if (fStatus == 0) {
      fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"));
      Assert(fStatus);
   }

   TString m;
   m.Form("%s::%s: %s", IsA()->GetName(), sub, mesg);
   fStatus->Add(m);
}

