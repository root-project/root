// @(#)root/proof:$Name:  $:$Id: TProofDraw.cxx,v 1.7 2005/03/10 17:57:04 rdm Exp $
// Author: Maarten Ballintijn, Marek Biskup  24/09/2003

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
#include "TH2F.h"
#include "TH3F.h"
#include "TProofDebug.h"
#include "TStatus.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TTree.h"
#include "TEventList.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TProofNTuple.h"
#include "TEnv.h"
#include "TNamed.h"
#include "TGraph.h"
#include "TPolyMarker3D.h"
#include "TVirtualPad.h"
#include "THLimitsFinder.h"
#include "TView.h"
#include "TStyle.h"


ClassImp(TProofDraw)

//______________________________________________________________________________
TProofDraw::TProofDraw()
   : fStatus(0), fManager(0)
{
   // Constructor.

   fVar[0]         = 0;
   fVar[1]         = 0;
   fVar[2]         = 0;
   fVar[3]         = 0;
   fManager        = 0;
   fMultiplicity   = 0;
   fSelect         = 0;
   fObjEval        = kFALSE;
   fDimension      = 0;
}


//______________________________________________________________________________
TProofDraw::~TProofDraw()
{
   // Destructor.

   ClearFormula();
}


//______________________________________________________________________________
void TProofDraw::Init(TTree *tree)
{
   // Init the tree.

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);
   fTree = tree;
   CompileVariables();
}


//______________________________________________________________________________
Bool_t TProofDraw::Notify()
{
   // Called when a new tree is loaded.

   PDB(kDraw,1) Info("Notify","Enter");
   if (fStatus == 0) {
      fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"));
      Assert(fStatus);
   }

   if (!fStatus->IsOk()) return kFALSE;

   if (fVar[0]) fVar[0]->UpdateFormulaLeaves();
   if (fVar[1]) fVar[1]->UpdateFormulaLeaves();
   if (fVar[2]) fVar[2]->UpdateFormulaLeaves();
   if (fVar[3]) fVar[3]->UpdateFormulaLeaves();
   if (fSelect) fSelect->UpdateFormulaLeaves();
   return kTRUE;
}


//______________________________________________________________________________
void TProofDraw::Begin(TTree *tree)
{
   // Executed by the client before processing.

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   if (fDrawInfo.GetObjectName() == "")
      fDrawInfo.SetObjectName("htemp");

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
   fTree = 0;
}


//______________________________________________________________________________
void TProofDraw::SlaveBegin(TTree* /*tree*/)
{
   // Executed by each slave before processing.
}


//______________________________________________________________________________
Bool_t TProofDraw::ProcessSingle(Long64_t entry, Int_t i)
{
   // Processes a single variable from an entry.

   Double_t w;
   Double_t v[4]; //[TDrawInfo::fgMaxDimension];

   if (fSelect)
      w = fSelect->EvalInstance(i);
   else
      w = 1.0;

   PDB(kDraw,3) Info("ProcessSingle","w[%d] = %f", i, w);

   if (w != 0.0) {
      Assert(fDimension <= TDrawInfo::fgMaxDimension);
      for (int j = 0; j < fDimension; j++)
         v[j] = fVar[j]->EvalInstance(i);
      if (fDimension >= 1);
         PDB(kDraw,4) Info("Process","v1[%d] = %f", i, v[0]);
      DoFill(entry, w, v);
   }
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TProofDraw::Process(Long64_t entry)
{
   // Executed for each entry.

   PDB(kDraw,3) Info("Process","Enter entry = %d", entry);

   fTree->LoadTree(entry);
   Int_t ndata = fManager->GetNdata();

   PDB(kDraw,3) Info("Process","ndata = %d", ndata);

   for (Int_t i=0;i<ndata;i++) {
      ProcessSingle(entry, i);
   }
   return kTRUE;
}


//______________________________________________________________________________
void TProofDraw::SlaveTerminate(void)
{
   // Executed by each slave after the processing has finished,
   // before returning the results to the client.

   PDB(kDraw,1) Info("SlaveTerminate","Enter");
}


//______________________________________________________________________________
void TProofDraw::Terminate(void)
{
   // Executed by the client after getting the processing retults.

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
}


//______________________________________________________________________________
void TProofDraw::ClearFormula()
{
   // Delete internal buffers.

   ResetBit(kWarn);
   SafeDelete(fVar[0]);
   SafeDelete(fVar[1]);
   SafeDelete(fVar[2]);
   SafeDelete(fVar[3]);
   SafeDelete(fSelect);
   fManager = 0;
   fMultiplicity = 0;
}


//______________________________________________________________________________
void TProofDraw::SetError(const char *sub, const char *mesg)
{
   // Sets the error status.

   if (fStatus == 0) {
      fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"));
      Assert(fStatus);
   }

   TString m;
   m.Form("%s::%s: %s", IsA()->GetName(), sub, mesg);
   fStatus->Add(m);
}


//______________________________________________________________________________
Bool_t TProofDraw::CompileVariables()
{
   // Compiles each variable from fDrawInfo for the tree fTree.
   // Return kFALSE if any of the variable is not compilable.

   fDimension = fDrawInfo.GetDimension();
   fMultiplicity = 0;
   fObjEval = kFALSE;
   if (strlen(fDrawInfo.GetSelection())) {
      fSelect = new TTreeFormula("Selection", fDrawInfo.GetSelection(), fTree);
      fSelect->SetQuickLoad(kTRUE);
      if (!fSelect->GetNdim()) {delete fSelect; fSelect = 0; return kFALSE; }
   }

   fManager = new TTreeFormulaManager();
   if (fSelect) fManager->Add(fSelect);
   fTree->ResetBit(TTree::kForceRead);

   for (int i = 0; i < fDimension; i++) {
      fVar[i] = new TTreeFormula(Form("Var%d", i),fDrawInfo.GetVarExp(i),fTree);
      fVar[i]->SetQuickLoad(kTRUE);
      if (!fVar[i]->GetNdim()) { ClearFormula(); return kFALSE;}
      fManager->Add(fVar[i]);
   }

   fManager->Sync();
   if (fManager->GetMultiplicity()==-1) fTree->SetBit(TTree::kForceRead);
   if (fManager->GetMultiplicity()>=1) fMultiplicity = fManager->GetMultiplicity();

   return kTRUE;

   if (fDimension==1) {
      TClass *cl = fVar[0]->EvalClass();
      if (cl) {
         fObjEval = kTRUE;
      }
   }
   return kTRUE;
}


ClassImp(TProofDrawHist)


//______________________________________________________________________________
void TProofDrawHist::Begin1D(TTree *)
{
   // Initialization for 1D Histogram.

   Assert(fDrawInfo.GetDimension() == 1);
   TObject* orig = fDrawInfo.GetOriginal();
   TH1* hold;
   if (fDrawInfo.GetNoParameters() == 0 && (hold = dynamic_cast<TH1*> (orig))) {
      TH1* hnew = (TH1*) hold->Clone();
      hnew->Reset();
      fInput->Add(hnew);
   } else {
      delete orig;
      fDrawInfo.SetOriginal(0);
      TString exp = fDrawInfo.GetVarExp();
      exp += ">>";
      double binsx, minx, maxx;
      if (fDrawInfo.IsSpecified(0))
         gEnv->SetValue("Hist.Binning.1D.x", fDrawInfo.GetParameter(0));
      binsx = gEnv->GetValue("Hist.Binning.1D.x",100);
      minx =  fDrawInfo.GetIfSpecified(1, 0);
      maxx =  fDrawInfo.GetIfSpecified(2, 0);
      exp += fDrawInfo.GetObjectName();
      exp += '(';
      exp +=      binsx;
      exp +=         ',';
      exp +=      minx;
      exp +=         ',';
      exp +=      maxx;
      exp += ')';

      fInitialExp = exp;
      TNamed *n = dynamic_cast<TNamed*> (fInput->FindObject("varexp"));
      if (n)
         n->SetTitle(exp);
      else
         Error("Begin", "Cannot find varexp on the fInput");
      if (fDrawInfo.GetNoParameters() != 3)
         fInput->Add(new TNamed("__PROOF_OPTIONS", "rebin"));
   }
}


//______________________________________________________________________________
void TProofDrawHist::Begin2D(TTree *)
{
   // Initialization for 2D histogram.

   Assert(fDrawInfo.GetDimension() == 2);
   TObject* orig = fDrawInfo.GetOriginal();
   TH2* hold;
   if (fDrawInfo.GetNoParameters() == 0 && (hold = dynamic_cast<TH2*> (orig))) {
      TH2* hnew = (TH2*) hold->Clone();
      hnew->Reset();
      fInput->Add(hnew);
   } else {
      delete orig;
      fDrawInfo.SetOriginal(0);
      TString exp = fDrawInfo.GetVarExp();
      exp += ">>";
      double binsx, minx, maxx;
      double binsy, miny, maxy;
      if (fDrawInfo.IsSpecified(0))
         gEnv->SetValue("Hist.Binning.2D.x", fDrawInfo.GetParameter(0));
      if (fDrawInfo.IsSpecified(3))
         gEnv->SetValue("Hist.Binning.2D.y", fDrawInfo.GetParameter(3));
      binsx = gEnv->GetValue("Hist.Binning.2D.x",100);
      minx =  fDrawInfo.GetIfSpecified(1, 0);
      maxx =  fDrawInfo.GetIfSpecified(2, 0);
      binsy = gEnv->GetValue("Hist.Binning.2D.y",100);
      miny =  fDrawInfo.GetIfSpecified(4, 0);
      maxy =  fDrawInfo.GetIfSpecified(5, 0);
      exp += fDrawInfo.GetObjectName();
      exp += '(';
      exp +=      binsx;
      exp +=         ',';
      exp +=      minx;
      exp +=         ',';
      exp +=      maxx;
      exp += ',';
      exp +=      binsy;
      exp +=         ',';
      exp +=      miny;
      exp +=         ',';
      exp +=      maxy;
      exp += ')';
      fInitialExp = exp;
      TNamed *n = dynamic_cast<TNamed*> (fInput->FindObject("varexp"));
      if (n)
         n->SetTitle(exp);
      else
         Error("Begin", "Cannot find varexp on the fInput");
      if (fDrawInfo.GetNoParameters() != 6)
         fInput->Add(new TNamed("__PROOF_OPTIONS", "rebin"));
   }
}


//______________________________________________________________________________
void TProofDrawHist::Begin3D(TTree *)
{
   // Initialization for 3D histogram.

   Assert(fDrawInfo.GetDimension() == 3);
   TObject* orig = fDrawInfo.GetOriginal();
   TH3* hold;
   if ((hold = dynamic_cast<TH3*> (orig)) && fDrawInfo.GetNoParameters() == 0) {
      TH3* hnew = (TH3*) hold->Clone();
      hnew->Reset();
      fInput->Add(hnew);
   } else {
      delete orig;
      fDrawInfo.SetOriginal(0);
      TString exp = fDrawInfo.GetVarExp();
      exp += ">>";
      double binsx, minx, maxx;
      double binsy, miny, maxy;
      double binsz, minz, maxz;
      if (fDrawInfo.IsSpecified(0))
         gEnv->SetValue("Hist.Binning.3D.x", fDrawInfo.GetParameter(0));
      if (fDrawInfo.IsSpecified(3))
         gEnv->SetValue("Hist.Binning.3D.y", fDrawInfo.GetParameter(3));
      if (fDrawInfo.IsSpecified(6))
         gEnv->SetValue("Hist.Binning.3D.z", fDrawInfo.GetParameter(6));
      binsx = gEnv->GetValue("Hist.Binning.3D.x",100);
      minx =  fDrawInfo.GetIfSpecified(1, 0);
      maxx =  fDrawInfo.GetIfSpecified(2, 0);
      binsy = gEnv->GetValue("Hist.Binning.3D.y",100);
      miny =  fDrawInfo.GetIfSpecified(4, 0);
      maxy =  fDrawInfo.GetIfSpecified(5, 0);
      binsz = gEnv->GetValue("Hist.Binning.3D.z",100);
      minz =  fDrawInfo.GetIfSpecified(7, 0);
      maxz =  fDrawInfo.GetIfSpecified(8, 0);
      exp += fDrawInfo.GetObjectName();
      exp += '(';
      exp +=      binsx;
      exp +=         ',';
      exp +=      minx;
      exp +=         ',';
      exp +=      maxx;
      exp += ',';
      exp +=      binsy;
      exp +=         ',';
      exp +=      miny;
      exp +=         ',';
      exp +=      maxy;
      exp += ',';
      exp +=      binsz;
      exp +=         ',';
      exp +=      minz;
      exp +=         ',';
      exp +=      maxz;
      exp += ')';
      fInitialExp = exp;
      TNamed *n = dynamic_cast<TNamed*> (fInput->FindObject("varexp"));
      if (n)
         n->SetTitle(exp);
      else
         Error("Begin", "Cannot find varexp on the fInput");
      if (fDrawInfo.GetNoParameters() != 9)
         fInput->Add(new TNamed("__PROOF_OPTIONS", "rebin"));
   }
}


//______________________________________________________________________________
void TProofDrawHist::Begin(TTree *tree)
{
   // See TProofDraw::Begin().

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   if (fDrawInfo.GetObjectName() == "")
      fDrawInfo.SetObjectName("htemp");

   switch (fDrawInfo.GetDimension()) {
      case 1:
         Begin1D(tree);
         break;
      case 2:
         Begin2D(tree);
         break;
      case 3:
         Begin3D(tree);
         break;
      default:
         Error("Begin", "Wrong dimension");
         break;
   }
   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
   fTree = 0;
}


//______________________________________________________________________________
void TProofDrawHist::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);
   if (fTree == 0) {
      if (!dynamic_cast<TH1*> (fDrawInfo.GetOriginal())) {
         fHistogram->SetLineColor(tree->GetLineColor());
         fHistogram->SetLineWidth(tree->GetLineWidth());
         fHistogram->SetLineStyle(tree->GetLineStyle());
         fHistogram->SetFillColor(tree->GetFillColor());
         fHistogram->SetFillStyle(tree->GetFillStyle());
         fHistogram->SetMarkerStyle(tree->GetMarkerStyle());
         fHistogram->SetMarkerColor(tree->GetMarkerColor());
         fHistogram->SetMarkerSize(tree->GetMarkerSize());
      }
   }
   fTree = tree;
   CompileVariables();
}


//______________________________________________________________________________
void TProofDrawHist::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   SafeDelete(fHistogram);

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   fDimension = fDrawInfo.GetDimension();
   TString exp = fDrawInfo.GetExp();
   if (fDrawInfo.GetOriginal()) {
      fHistogram = dynamic_cast<TH1*> (fDrawInfo.GetOriginal());
      if (fHistogram) {
         fOutput->Add(fHistogram);
         PDB(kDraw,1) Info("SlaveBegin","Original histogram found");
         return;
      }
      else
         Error("SlaveBegin","Original object found but it is not a histogram");
   }

   Int_t countx = 100; double minx = 0, maxx = 0;
   Int_t county = 100; double miny = 0, maxy = 0;
   Int_t countz = 100; double minz = 0, maxz = 0;
   if (fDrawInfo.GetNoParameters() != 0) {
      countx = (Int_t) fDrawInfo.GetIfSpecified(0, countx);
      county = (Int_t) fDrawInfo.GetIfSpecified(3, county);
      countz = (Int_t) fDrawInfo.GetIfSpecified(6, countz);
      minx =  fDrawInfo.GetIfSpecified(1, minx);
      maxx =  fDrawInfo.GetIfSpecified(2, maxx);
      miny =  fDrawInfo.GetIfSpecified(4, miny);
      maxy =  fDrawInfo.GetIfSpecified(5, maxy);
      minz =  fDrawInfo.GetIfSpecified(7, minz);
      maxz =  fDrawInfo.GetIfSpecified(8, maxz);
   }
   if (fDrawInfo.GetNoParameters() != 3*fDimension)
      Error("SlaveBegin", "Impossible - Wrong number of parameters");

   if (fDimension == 1)
      fHistogram = new TH1F(fDrawInfo.GetObjectName(),
                            fDrawInfo.GetObjectTitle(),
                            countx, minx, maxx);
   else if (fDimension == 2){
      fHistogram = new TH2F(fDrawInfo.GetObjectName(),
                            fDrawInfo.GetObjectTitle(),
                            countx, minx, maxx,
                            county, miny, maxy);
   }
   else if (fDimension == 3) {
      fHistogram = new TH3F(fDrawInfo.GetObjectName(),
                            fDrawInfo.GetObjectTitle(),
                            countx, minx, maxx,
                            county, miny, maxy,
                            countz, minz, maxz);
   } else {
      Info("Begin", "Wrong dimension");
      return;        // FIXME: end the session
   }
   if (minx >= maxx)
      fHistogram->SetBuffer(TH1::GetDefaultBufferSize());
   if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("__PROOF_OPTIONS"))) {
      if (strstr(opt->GetTitle(), "rebin"))
         fHistogram->SetBit(TH1::kCanRebin);
   }
   fHistogram->SetDirectory(0);   // take ownership
   fOutput->Add(fHistogram);      // release ownership

   fTree = 0;
   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawHist::DoFill(Long64_t, Double_t w, const Double_t *v)
{
   // Fills the histgram with given values.

   if (fDimension == 1)
      fHistogram->Fill(v[0], w);
   else if (fDimension == 2)
      ((TH2F *)fHistogram)->Fill(v[1], v[0], w);
   else if (fDimension == 3)
      ((TH3F *)fHistogram)->Fill(v[2], v[1], v[0], w);
}


//______________________________________________________________________________
void TProofDrawHist::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fHistogram = (TH1F *) fOutput->FindObject(fDrawInfo.GetObjectName());
   if (fHistogram) {
      SetStatus((Int_t) fHistogram->GetEntries());
      if (TH1* old = dynamic_cast<TH1*> (fDrawInfo.GetOriginal())) {
         if (!fDrawInfo.GetAdd())
            old->Reset();
         TList l;
         l.Add(fHistogram);
         old->Merge(&l);
         fOutput->Remove(fHistogram);
         delete fHistogram;
         if (fDrawInfo.GetDraw())
            old->Draw(fOption.Data());
      } else {
         if (fDrawInfo.GetDraw())
            fHistogram->Draw(fOption.Data());
         fHistogram->SetTitle(fDrawInfo.GetObjectTitle());
      }
   }
}


ClassImp(TProofDrawEventList)

//______________________________________________________________________________
TProofDrawEventList::~TProofDrawEventList()
{
   // Destructor.

   SafeDelete(fElist);
   SafeDelete(fEventLists);
}


//______________________________________________________________________________
void TProofDrawEventList::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);

   if (fTree) {      // new tree is being set
      if (!fElist)
         Error("Init", "Impossible - fElist cannot be 0");
      fEventLists->Add(fElist);
   }
   fElist = new TEventList(tree->GetDirectory()->GetName(), tree->GetName());
   fTree = tree;
   CompileVariables();
}


//______________________________________________________________________________
void TProofDrawEventList::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);

   SafeDelete(fEventLists);

   fDimension = 0;
   fTree = 0;
   fEventLists = new TList();
   fEventLists->SetName("_PROOF_EventListsList");
   fOutput->Add(fEventLists);

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawEventList::DoFill(Long64_t entry, Double_t , const Double_t *)
{
   // Fills the eventlist with given values.

   fElist->Enter(entry);
}


//______________________________________________________________________________
void TProofDrawEventList::SlaveTerminate(void)
{
   // See TProofDraw::SlaveTerminate().

   PDB(kDraw,1) Info("SlaveTerminate","Enter");
   fEventLists->Add(fElist);
   fEventLists = 0;
   fElist = 0;
}


//______________________________________________________________________________
void TProofDrawEventList::Terminate(void)
{
   // See TProofDraw::Terminate().

   TProofDraw::Terminate();   // take care of fStatus
   if (!fStatus)
      return;

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);

   TEventList *el = dynamic_cast<TEventList*> (fOutput->FindObject("_PROOF_EventList"));
   if (el) {
      el->SetName(fInitialExp.Data()+2);
      SetStatus(el->GetN());
      if (TEventList* old = dynamic_cast<TEventList*> (fDrawInfo.GetOriginal())) {
         if (!fDrawInfo.GetAdd())
            old->Reset();
         old->Add(el);
         fOutput->Remove(el);
         delete el;
      }
   }
   else
      Error("Terminate", "Cannot find output EventList");

}


ClassImp(TProofDrawProfile)


//______________________________________________________________________________
void TProofDrawProfile::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);

   if (fTree == 0) {
      if (!dynamic_cast<TProfile*> (fDrawInfo.GetOriginal())) {
         fProfile->SetLineColor(tree->GetLineColor());
         fProfile->SetLineWidth(tree->GetLineWidth());
         fProfile->SetLineStyle(tree->GetLineStyle());
         fProfile->SetFillColor(tree->GetFillColor());
         fProfile->SetFillStyle(tree->GetFillStyle());
         fProfile->SetMarkerStyle(tree->GetMarkerStyle());
         fProfile->SetMarkerColor(tree->GetMarkerColor());
         fProfile->SetMarkerSize(tree->GetMarkerSize());
      }
   }
   fTree = tree;
   CompileVariables();
}


//______________________________________________________________________________
void TProofDrawProfile::Begin(TTree *tree)
{
   // See TProofDraw::Begin().

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);

   Assert(fDrawInfo.GetDimension() == 2);

   TObject *orig = fDrawInfo.GetOriginal();
   TH1* pold;
   if ((pold = dynamic_cast<TProfile*> (orig)) && fDrawInfo.GetNoParameters() == 0) {
      TProfile* pnew = (TProfile*) pold->Clone();
      pnew->Reset();
      fInput->Add(pnew);
   } else {
      delete orig;
      fDrawInfo.SetOriginal(0);
      TString exp = fDrawInfo.GetVarExp();
      exp += ">>";
      double binsx, minx, maxx;
      if (fDrawInfo.IsSpecified(0))
         gEnv->SetValue("Hist.Binning.2D.Prof", fDrawInfo.GetParameter(0));
      binsx = gEnv->GetValue("Hist.Binning.2D.Prof",100);
      minx =  fDrawInfo.GetIfSpecified(1, 0);
      maxx =  fDrawInfo.GetIfSpecified(2, 0);
      if (fDrawInfo.GetObjectName() == "")
         fDrawInfo.SetObjectName("htemp");
      exp += fDrawInfo.GetObjectName();
      exp += '(';
      exp +=      binsx;
      exp +=         ',';
      exp +=      minx;
      exp +=         ',';
      exp +=      maxx;
      exp += ')';
      fInitialExp = exp;
      TNamed *n = dynamic_cast<TNamed*> (fInput->FindObject("varexp"));
      if (n)
         n->SetTitle(exp);
      else
         Error("Begin", "Cannot find varexp on the fInput");
      if (fDrawInfo.GetNoParameters() != 3)
         fInput->Add(new TNamed("__PROOF_OPTIONS", "rebin"));
   }

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
   fTree = 0;
}


//______________________________________________________________________________
void TProofDrawProfile::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   SafeDelete(fProfile);


   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   fDimension = 2;
   TString exp = fDrawInfo.GetExp();

   if (fDrawInfo.GetOriginal()) {
      fProfile = dynamic_cast<TProfile*> (fDrawInfo.GetOriginal());
      if (fProfile) {
         fOutput->Add(fProfile);
         PDB(kDraw,1) Info("SlaveBegin","Original profile histogram found");
         return;
      }
      else
         Error("SlaveBegin","Original object found but it is not a histogram");
   }
   Int_t countx = 100; double minx = 0, maxx = 0;
   if (fDrawInfo.GetNoParameters() != 0) {
      countx = (Int_t) fDrawInfo.GetIfSpecified(0, countx);
      minx =  fDrawInfo.GetIfSpecified(1, minx);
      maxx =  fDrawInfo.GetIfSpecified(2, maxx);
   }
   if (fDrawInfo.GetNoParameters() != 3)
      Error("SlaveBegin", "Impossible - Wrong number of parameters");
   TString constructorOptions = "";
   if (fOption.Contains("profs"))
      constructorOptions = "s";
   fProfile = new TProfile(fDrawInfo.GetObjectName(),
                           fDrawInfo.GetObjectTitle(),
                           countx, minx, maxx,
                           constructorOptions);
   if (minx >= maxx)
      fProfile->SetBuffer(TH1::GetDefaultBufferSize());

   if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("__PROOF_OPTIONS"))) {
      if (strstr(opt->GetTitle(), "rebin"))
         fProfile->SetBit(TH1::kCanRebin);
   }
   fProfile->SetDirectory(0);   // take ownership
   fOutput->Add(fProfile);      // release ownership
   fTree = 0;
   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawProfile::DoFill(Long64_t , Double_t w, const Double_t *v)
{
   // Fills the profile histogram with the given values.

   fProfile->Fill(v[1], v[0], w);
}


//______________________________________________________________________________
void TProofDrawProfile::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fProfile = (TProfile *) fOutput->FindObject(fDrawInfo.GetObjectName());
   if (fProfile) {
      SetStatus((Int_t) fProfile->GetEntries());
      if (TProfile* old = dynamic_cast<TProfile*> (fDrawInfo.GetOriginal())) {
         if (!fDrawInfo.GetAdd())
            old->Reset();
         TList l;
         l.Add(fProfile);
         old->Merge(&l);
         fOutput->Remove(fProfile);
         delete fProfile;
         if (fDrawInfo.GetDraw())
            old->Draw(fOption.Data());
      } else {
         if (fDrawInfo.GetDraw())
            fProfile->Draw(fOption.Data());
         fProfile->SetTitle(fDrawInfo.GetObjectTitle());
      }
   }
}


ClassImp(TProofDrawProfile2D)

//______________________________________________________________________________
void TProofDrawProfile2D::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);
   if (fTree == 0) {
      if (!dynamic_cast<TProfile2D*> (fDrawInfo.GetOriginal())) {
         fProfile->SetLineColor(tree->GetLineColor());
         fProfile->SetLineWidth(tree->GetLineWidth());
         fProfile->SetLineStyle(tree->GetLineStyle());
         fProfile->SetFillColor(tree->GetFillColor());
         fProfile->SetFillStyle(tree->GetFillStyle());
         fProfile->SetMarkerStyle(tree->GetMarkerStyle());
         fProfile->SetMarkerColor(tree->GetMarkerColor());
         fProfile->SetMarkerSize(tree->GetMarkerSize());
      }
   }

   fTree = tree;
   CompileVariables();
}


//______________________________________________________________________________
void TProofDrawProfile2D::Begin(TTree *tree)
{
   // See TProofDraw::Begin().

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);

   Assert(fDrawInfo.GetDimension() == 3);

   TObject *orig = fDrawInfo.GetOriginal();
   TProfile2D *pold;
   if ((pold = dynamic_cast<TProfile2D*> (orig)) && fDrawInfo.GetNoParameters() == 0) {
      TProfile2D* pnew = (TProfile2D*) pold->Clone();
      pnew->Reset();
      fInput->Add(pnew);
   } else {
      delete orig;
      fDrawInfo.SetOriginal(0);
      TString exp = fDrawInfo.GetVarExp();
      exp += ">>";
      double binsx, minx, maxx;
      double binsy, miny, maxy;
      if (fDrawInfo.IsSpecified(0))
         gEnv->SetValue("Hist.Binning.3D.Profx", fDrawInfo.GetParameter(0));
      if (fDrawInfo.IsSpecified(3))
         gEnv->SetValue("Hist.Binning.3D.Profy", fDrawInfo.GetParameter(3));
      binsx = gEnv->GetValue("Hist.Binning.3D.Profx",20);
      minx =  fDrawInfo.GetIfSpecified(1, 0);
      maxx =  fDrawInfo.GetIfSpecified(2, 0);
      binsy = gEnv->GetValue("Hist.Binning.3D.Profy",20);
      miny =  fDrawInfo.GetIfSpecified(4, 0);
      maxy =  fDrawInfo.GetIfSpecified(5, 0);
      if (fDrawInfo.GetObjectName() == "")
         fDrawInfo.SetObjectName("htemp");
      exp += fDrawInfo.GetObjectName();
      exp += '(';
      exp +=      binsx;
      exp +=         ',';
      exp +=      minx;
      exp +=         ',';
      exp +=      maxx;
      exp += ',';
      exp +=      binsy;
      exp +=         ',';
      exp +=      miny;
      exp +=         ',';
      exp +=      maxy;
      exp += ')';
      fInitialExp = exp;
      TNamed *n = dynamic_cast<TNamed*> (fInput->FindObject("varexp"));
      if (n)
         n->SetTitle(exp);
      else
         Error("Begin", "Cannot find varexp on the fInput");
      if (fDrawInfo.GetNoParameters() != 6)
         fInput->Add(new TNamed("__PROOF_OPTIONS", "rebin"));
   }

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawProfile2D::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   SafeDelete(fProfile);

   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   fDimension = 2;
   TString exp = fDrawInfo.GetExp();

   if (fDrawInfo.GetOriginal()) {
      fProfile = dynamic_cast<TProfile2D*> (fDrawInfo.GetOriginal());
      if (fProfile) {
         fOutput->Add(fProfile);
         PDB(kDraw,1) Info("SlaveBegin","Original profile histogram found");
         return;
      } else
         Error("SlaveBegin","Original object found but it is not a histogram");
   }
   Int_t countx = 40; double minx = 0, maxx = 0;
   Int_t county = 40; double miny = 0, maxy = 0;
   if (fDrawInfo.GetNoParameters() != 0) {
      countx = (Int_t) fDrawInfo.GetIfSpecified(0, countx);
      minx =  fDrawInfo.GetIfSpecified(1, minx);
      maxx =  fDrawInfo.GetIfSpecified(2, maxx);
      county = (Int_t) fDrawInfo.GetIfSpecified(3, countx);
      miny =  fDrawInfo.GetIfSpecified(4, minx);
      maxy =  fDrawInfo.GetIfSpecified(5, maxx);
   }
   if (fDrawInfo.GetNoParameters() != 6)
      Error("SlaveBegin", "Impossible - Wrong number of parameters");
   TString constructorOptions = "";
   if (fOption.Contains("profs"))
      constructorOptions = "s";
   fProfile = new TProfile2D(fDrawInfo.GetObjectName(),
                             fDrawInfo.GetObjectTitle(),
                             countx, minx, maxx,
                             county, miny, maxy,
                             constructorOptions);
   if (minx >= maxx)
      fProfile->SetBuffer(TH1::GetDefaultBufferSize());

   if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("__PROOF_OPTIONS"))) {
      if (strstr(opt->GetTitle(), "rebin"))
         fProfile->SetBit(TH1::kCanRebin);
   }
   fProfile->SetDirectory(0);   // take ownership
   fOutput->Add(fProfile);      // release ownership
   fTree = 0;
   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawProfile2D::DoFill(Long64_t , Double_t w, const Double_t *v)
{
   // Fills the histogram with the given values.

   fProfile->Fill(v[2], v[1], v[0], w);
}


//______________________________________________________________________________
void TProofDrawProfile2D::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fProfile = (TProfile2D *) fOutput->FindObject(fDrawInfo.GetObjectName());
   if (fProfile) {
      SetStatus((Int_t) fProfile->GetEntries());
      if (TProfile2D* old = dynamic_cast<TProfile2D*> (fDrawInfo.GetOriginal())) {
         if (!fDrawInfo.GetAdd())
            old->Reset();
         TList l;
         l.Add(fProfile);
         old->Merge(&l);
         fOutput->Remove(fProfile);
         delete fProfile;
         if (fDrawInfo.GetDraw())
            old->Draw(fOption.Data());
      } else {
         if (fDrawInfo.GetDraw())
            fProfile->Draw(fOption.Data());
         fProfile->SetTitle(fDrawInfo.GetObjectTitle());
      }
   }
}


ClassImp(TProofDrawGraph)

//______________________________________________________________________________
void TProofDrawGraph::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fDrawInfo.Parse(fInitialExp, fSelection, fOption);

   SafeDelete(fScatterPlot);
   fDimension = 2;

   fScatterPlot = new TProofNTuple(2);
   fScatterPlot->SetName("__PROOF_SCATTERPLOT");
   fOutput->Add(fScatterPlot);      // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawGraph::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the graph with the given values.

   fScatterPlot->Fill(v[1], v[0]);
}


//______________________________________________________________________________
void TProofDrawGraph::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fScatterPlot = (TProofNTuple*) fOutput->FindObject("__PROOF_SCATTERPLOT");
   if (fScatterPlot) {
      SetStatus((Int_t) fScatterPlot->GetEntries());
      TH2F* hist;
      TObject *orig = fDrawInfo.GetOriginal();
      if ( (hist = dynamic_cast<TH2F*> (orig)) == 0 ) {
         delete orig;
         fDrawInfo.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         if (fDrawInfo.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.2D.x", fDrawInfo.GetParameter(0));
         if (fDrawInfo.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.2D.y", fDrawInfo.GetParameter(3));
         binsx = gEnv->GetValue("Hist.Binning.2D.x",100);
         minx =  fDrawInfo.GetIfSpecified(1, 0);
         maxx =  fDrawInfo.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.2D.y",100);
         miny =  fDrawInfo.GetIfSpecified(4, 0);
         maxy =  fDrawInfo.GetIfSpecified(5, 0);
         hist = new TH2F(fDrawInfo.GetObjectName(), fDrawInfo.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx, (Int_t) binsy, miny, maxy);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fDrawInfo.GetNoParameters() != 6)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);
//         if (fDrawInfo.GetDraw())    // ?? FIXME
//            hist->SetDirectory(0);
      } else {
         if (!fDrawInfo.GetAdd())
            hist->Reset();
      }
      if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
         Double_t xmin = fScatterPlot->Min(1);
         Double_t xmax = fScatterPlot->Max(1);
         Double_t ymin = fScatterPlot->Min(2);
         Double_t ymax = fScatterPlot->Max(2);
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,xmin,xmax,ymin,ymax);
      }
      if (!hist->TestBit(kCanDelete)) {
         TH1 *h2c = hist->DrawCopy(fOption.Data());
         h2c->SetStats(kFALSE);
      } else {
         hist->Draw();
      }
      gPad->Update();

      TGraph *g = new TGraph(fScatterPlot->GetEntries());

      for (int i = 0; i < fScatterPlot->GetEntries(); i++)
         g->SetPoint(i, fScatterPlot->GetX(i), fScatterPlot->GetY(i));

      g->SetEditable(kFALSE);
      g->SetBit(kCanDelete);
      // FIXME set color, marker size, etc.

      if (fDrawInfo.GetDraw()) {
         if (fOption == "" || strcmp(fOption, "same") == 0)
            g->Draw("p");
         else
            g->Draw(fOption);
         gPad->Update();
      }
      if (!hist->TestBit(kCanDelete)) {
         for (int i = 0; i < fScatterPlot->GetEntries(); i++)
            hist->Fill(fScatterPlot->GetX(i), fScatterPlot->GetY(i), 1);
      }
   }
}


ClassImp(TProofDrawPolyMarker3D)

//______________________________________________________________________________
void TProofDrawPolyMarker3D::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   Assert(fDrawInfo.GetDimension() == 3);

   SafeDelete(fScatterPlot);
   fDimension = 3;

   fScatterPlot = new TProofNTuple(3);
   fScatterPlot->SetName("__PROOF_SCATTERPLOT");
   fOutput->Add(fScatterPlot);      // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawPolyMarker3D::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the scatter plot with the given values.

   fScatterPlot->Fill(v[2], v[1], v[0]);
}


//______________________________________________________________________________
void TProofDrawPolyMarker3D::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fScatterPlot = (TProofNTuple*) fOutput->FindObject("__PROOF_SCATTERPLOT");
   if (fScatterPlot) {
      SetStatus((Int_t) fScatterPlot->GetEntries());
      TH3F* hist;
      TObject *orig = fDrawInfo.GetOriginal();
      if ( (hist = dynamic_cast<TH3F*> (orig)) == 0 ) {
         delete orig;
         fDrawInfo.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         double binsz, minz, maxz;
         if (fDrawInfo.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.3D.x", fDrawInfo.GetParameter(0));
         if (fDrawInfo.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.3D.y", fDrawInfo.GetParameter(3));
         if (fDrawInfo.IsSpecified(6))
            gEnv->SetValue("Hist.Binning.3D.z", fDrawInfo.GetParameter(6));
         binsx = gEnv->GetValue("Hist.Binning.3D.x",100);
         minx =  fDrawInfo.GetIfSpecified(1, 0);
         maxx =  fDrawInfo.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.3D.y",100);
         miny =  fDrawInfo.GetIfSpecified(4, 0);
         maxy =  fDrawInfo.GetIfSpecified(5, 0);
         binsz = gEnv->GetValue("Hist.Binning.3D.z",100);
         minz =  fDrawInfo.GetIfSpecified(7, 0);
         maxz =  fDrawInfo.GetIfSpecified(8, 0);
         hist = new TH3F(fDrawInfo.GetObjectName(), fDrawInfo.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx,
                        (Int_t) binsy, miny, maxy,
                        (Int_t) binsz, minz, maxz);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fDrawInfo.GetNoParameters() != 9)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);
//         if (fDrawInfo.GetDraw())    // ?? FIXME
//            hist->SetDirectory(0);
      } else {
         if (!fDrawInfo.GetAdd())
            hist->Reset();
      }

      Double_t rmin[3], rmax[3];

      // FIXME take rmin and rmax from the old histogram
      if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
         rmin[0] = fScatterPlot->Min(1);
         rmax[0] = fScatterPlot->Max(1);
         rmin[1] = fScatterPlot->Min(2);
         rmax[1] = fScatterPlot->Max(2);
         rmin[2] = fScatterPlot->Min(3);
         rmax[2] = fScatterPlot->Max(3);
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,
                           rmin[0], rmax[0], rmin[1], rmax[1], rmin[2], rmax[2]);
      }
      if (fDrawInfo.GetDraw()) {
         if (!hist->TestBit(kCanDelete)) {
            TH1 *histcopy = hist->DrawCopy(fOption.Data());
            histcopy->SetStats(kFALSE);
         }
         else
            hist->Draw();        // no draw options on purpose
         gPad->Update();
      } else {
         gPad->Clear();
         gPad->Range(-1,-1,1,1);
         new TView(rmin,rmax,1);
      }
      TPolyMarker3D *pm3D = new TPolyMarker3D(fScatterPlot->GetEntries());
      // FIXME set marker style
      for (int i = 0; i < fScatterPlot->GetEntries(); i++)
         pm3D->SetPoint(i, fScatterPlot->GetX(i), fScatterPlot->GetY(i), fScatterPlot->GetZ(i));
      if (fDrawInfo.GetDraw())
         pm3D->Draw(fOption);
      gPad->Update();
      if (!hist->TestBit(kCanDelete)) {
         for (int i = 0; i < fScatterPlot->GetEntries(); i++)
            hist->Fill(fScatterPlot->GetX(i), fScatterPlot->GetY(i), fScatterPlot->GetZ(i), 1);
      }
   }
}


ClassImp(TProofDrawListOfGraphs)

//______________________________________________________________________________
void TProofDrawListOfGraphs::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   Assert(fDrawInfo.GetDimension() == 3);

   SafeDelete(fScatterPlot);
   fDimension = 3;

   fScatterPlot = new TProofNTuple(3);
   fScatterPlot->SetName("__PROOF_SCATTERPLOT");
   fOutput->Add(fScatterPlot);      // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawListOfGraphs::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the scatter plot with the given values.

   fScatterPlot->Fill(v[2], v[1], v[0]);
}


//______________________________________________________________________________
void TProofDrawListOfGraphs::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fScatterPlot = (TProofNTuple*) fOutput->FindObject("__PROOF_SCATTERPLOT");
   if (fScatterPlot) {
      SetStatus((Int_t) fScatterPlot->GetEntries());
      TH2F* hist;
      TObject *orig = fDrawInfo.GetOriginal();
      if ( (hist = dynamic_cast<TH2F*> (orig)) == 0 ) {
         delete orig;
         fDrawInfo.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         if (fDrawInfo.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.2D.x", fDrawInfo.GetParameter(0));
         if (fDrawInfo.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.2D.y", fDrawInfo.GetParameter(3));
         binsx = gEnv->GetValue("Hist.Binning.2D.x", 40);
         minx =  fDrawInfo.GetIfSpecified(1, 0);
         maxx =  fDrawInfo.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.2D.y", 40);
         miny =  fDrawInfo.GetIfSpecified(4, 0);
         maxy =  fDrawInfo.GetIfSpecified(5, 0);
         hist = new TH2F(fDrawInfo.GetObjectName(), fDrawInfo.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx, (Int_t) binsy, miny, maxy);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fDrawInfo.GetNoParameters() != 6)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);

//         if (fDrawInfo.GetDraw())         // ?? FIXME
//            hist->SetDirectory(0);
      }
      Double_t rmin[3], rmax[3];

      // FIXME take rmin and rmax from the old histogram
      rmin[0] = fScatterPlot->Min(1);
      rmax[0] = fScatterPlot->Max(1);
      rmin[1] = fScatterPlot->Min(2);
      rmax[1] = fScatterPlot->Max(2);
      rmin[2] = fScatterPlot->Min(3);
      rmax[2] = fScatterPlot->Max(3);
      // in this case we don't care about user-specified limits
      if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,
                        rmin[1], rmax[1], rmin[2], rmax[2]);
      }

      Int_t ncolors  = gStyle->GetNumberOfColors();
      TObjArray *grs = (TObjArray*)hist->GetListOfFunctions()->FindObject("graphs");
      Int_t col;
      TGraph *gr;
      if (!grs) {
         grs = new TObjArray(ncolors);
         grs->SetOwner();
         grs->SetName("graphs");
         hist->GetListOfFunctions()->Add(grs, "P");
         for (col=0;col<ncolors;col++) {
            gr = new TGraph();
            gr->SetMarkerColor(col);
//            gr->SetMarkerStyle(fTree->GetMarkerStyle());
//            gr->SetMarkerSize(fTree->GetMarkerSize());
            grs->AddAt(gr,col);
         }
      }
      // Fill the graphs acording to the color
      for (int i=0;i<fScatterPlot->GetEntries();i++) {
         col = Int_t((ncolors-1)*((fScatterPlot->GetX(i)-rmin[0])/(rmax[0]-rmin[0])));
         if (col < 0) col = 0;
         if (col > ncolors-1) col = ncolors-1;
         gr = (TGraph*)grs->UncheckedAt(col);
         if (gr) gr->SetPoint(gr->GetN(), fScatterPlot->GetY(i), fScatterPlot->GetZ(i));
      }
      // Remove potential empty graphs
      for (col=0;col<ncolors;col++) {
         gr = (TGraph*)grs->At(col);
         if (gr && gr->GetN() <= 0) grs->Remove(gr);
      }
      if (fDrawInfo.GetDraw()) {
         hist->Draw(fOption.Data());
         gPad->Update();
      }
   }
}


ClassImp(TProofDrawListOfPolyMarkers3D)


//______________________________________________________________________________
void TProofDrawListOfPolyMarkers3D::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fDrawInfo.Parse(fInitialExp, fSelection, fOption);
   Assert(fDrawInfo.GetDimension() == 4);

   SafeDelete(fScatterPlot);
   fDimension = 4;

   fScatterPlot = new TProofNTuple(4);
   fScatterPlot->SetName("__PROOF_SCATTERPLOT");
   fOutput->Add(fScatterPlot);      // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawListOfPolyMarkers3D::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the scatter plot with the given values.

   fScatterPlot->Fill(v[3], v[2], v[1], v[0]);
}


//______________________________________________________________________________
void TProofDrawListOfPolyMarkers3D::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fScatterPlot = (TProofNTuple*) fOutput->FindObject("__PROOF_SCATTERPLOT");
   if (fScatterPlot) {
      SetStatus((Int_t) fScatterPlot->GetEntries());
      TH3F* hist;
      TObject *orig = fDrawInfo.GetOriginal();
      if ( (hist = dynamic_cast<TH3F*> (orig)) == 0 || fDrawInfo.GetNoParameters() != 0) {
         delete orig;
         fDrawInfo.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         double binsz, minz, maxz;
         if (fDrawInfo.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.3D.x", fDrawInfo.GetParameter(0));
         if (fDrawInfo.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.3D.y", fDrawInfo.GetParameter(3));
         if (fDrawInfo.IsSpecified(6))
            gEnv->SetValue("Hist.Binning.3D.z", fDrawInfo.GetParameter(3));
         binsx = gEnv->GetValue("Hist.Binning.3D.x", 20);
         minx =  fDrawInfo.GetIfSpecified(1, 0);
         maxx =  fDrawInfo.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.3D.y", 20);
         miny =  fDrawInfo.GetIfSpecified(4, 0);
         maxy =  fDrawInfo.GetIfSpecified(5, 0);
         binsz = gEnv->GetValue("Hist.Binning.3D.z", 20);
         minz =  fDrawInfo.GetIfSpecified(7, 0);
         maxz =  fDrawInfo.GetIfSpecified(8, 0);
         hist = new TH3F(fDrawInfo.GetObjectName(), fDrawInfo.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx,
                        (Int_t) binsy, miny, maxy,
                        (Int_t) binsz, minz, maxz);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fDrawInfo.GetNoParameters() != 9)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);

//         if (fDrawInfo.GetDraw())          // ?? FIXME
//            hist->SetDirectory(0);
      }
      Double_t rmin[4], rmax[4];

      // FIXME take rmin and rmax from the old histogram
      if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
         rmin[0] = fScatterPlot->Min(1);
         rmax[0] = fScatterPlot->Max(1);
         rmin[1] = fScatterPlot->Min(2);
         rmax[1] = fScatterPlot->Max(2);
         rmin[2] = fScatterPlot->Min(3);
         rmax[2] = fScatterPlot->Max(3);
         rmin[3] = fScatterPlot->Min(4);
         rmax[3] = fScatterPlot->Max(4);
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,
                           rmin[1], rmax[1], rmin[2], rmax[2], rmin[3], rmax[3]);
      }
      Int_t ncolors  = gStyle->GetNumberOfColors();
      TObjArray *pms = (TObjArray*)hist->GetListOfFunctions()->FindObject("polymarkers");
      Int_t col;
      TPolyMarker3D *pm3d;
      if (!pms) {
         pms = new TObjArray(ncolors);
         pms->SetOwner();
         pms->SetName("polymarkers");
         hist->GetListOfFunctions()->Add(pms);
         for (col=0;col<ncolors;col++) {
            pm3d = new TPolyMarker3D();
            pm3d->SetMarkerColor(col);
//            pm3d->SetMarkerStyle(fTree->GetMarkerStyle());
//            pm3d->SetMarkerSize(fTree->GetMarkerSize());
            pms->AddAt(pm3d,col);
         }
      }
      for (Int_t i=0;i<fScatterPlot->GetEntries();i++) {
         col = Int_t(fScatterPlot->GetX(i));
         if (col < 0) col = 0;
         if (col > ncolors-1) col = ncolors-1;
         pm3d = (TPolyMarker3D*)pms->UncheckedAt(col);
         pm3d->SetPoint(pm3d->GetLastPoint()+1,
            fScatterPlot->GetY(i),
            fScatterPlot->GetZ(i),
            fScatterPlot->GetT(i));
      }
      if (fDrawInfo.GetDraw()) {
         hist->Draw(fOption.Data());
         gPad->Update();
      }
   }
}

