// @(#)root/proofplayer:$Id$
// Author: Maarten Ballintijn, Marek Biskup  24/09/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofDraw                                                           //
//                                                                      //
// Implement Tree drawing using PROOF.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TProofDraw.h"
#include "TAttFill.h"
#include "TAttLine.h"
#include "TAttMarker.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TError.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TProof.h"
#include "TProofDebug.h"
#include "TStatus.h"
#include "TTreeDrawArgsParser.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TTree.h"
#include "TEventList.h"
#include "TEntryList.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TEnv.h"
#include "TNamed.h"
#include "TGraph.h"
#include "TPolyMarker3D.h"
#include "TVirtualPad.h"
#include "THLimitsFinder.h"
#include "TView.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TROOT.h"

#include <algorithm>
using namespace std;


// Simple call to draw a canvas on the fly from applications loading
// this plug-in dynamically
extern "C" {
   Int_t DrawCanvas(TObject *obj)
   {
      // Draw the object if deriving from a canvas

      if (TCanvas* c = dynamic_cast<TCanvas *> (obj)) {
         c->Draw();
         return 0;
      }
      // Not a TCanvas
      return 1;
   }
}

// Simple call to parse arguments on the fly from applications loading
// this plug-in dynamically
extern "C" {
   Int_t GetDrawArgs(const char *var, const char *sel, Option_t *opt,
                     TString &selector, TString &objname)
   {
      // Parse arguments with the help of TTreeDrawArgsParser

      TTreeDrawArgsParser info;
      info.Parse(var, sel, opt);
      selector = info.GetProofSelectorName();
      objname = info.GetObjectName();

      // Done
      return 0;
   }
}

// Simple call to create destroy a 'named' canvas
extern "C" {
   void FeedBackCanvas(const char *name, Bool_t create)
   {
      // Create or destroy canvas 'name'

      if (create) {
         new TCanvas(name, "FeedBack", 800,30,700,500);
      } else {
         TCanvas *c = (TCanvas *) gROOT->GetListOfCanvases()->FindObject(name);
         if (c) delete c;
      }
      // Done
      return;
   }
}

ClassImp(TProofDraw)

//______________________________________________________________________________
TProofDraw::TProofDraw()
   : fStatus(0), fManager(0), fTree(0)
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
   fWeight         = 1.;
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
      if (!(fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"))))
         return kFALSE;
   }
   if (!fStatus->IsOk()) return kFALSE;
   if (!fManager) return kFALSE;
   fManager->UpdateFormulaLeaves();
   return kTRUE;
}

//______________________________________________________________________________
void TProofDraw::Begin(TTree *tree)
{
   // Executed by the client before processing.

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   if (fTreeDrawArgsParser.GetObjectName() == "")
      fTreeDrawArgsParser.SetObjectName("htemp");

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
   fTree = 0;
}


//______________________________________________________________________________
void TProofDraw::SlaveBegin(TTree* /*tree*/)
{
   // Executed by each slave before processing.

   // Get the weight
   TProofDraw::FillWeight();
}

//______________________________________________________________________________
void TProofDraw::FillWeight()
{
   // Get weight from input list, if any

   Double_t ww;
   if (TProof::GetParameter(fInput, "PROOF_ChainWeight", ww) == 0)
      fWeight = ww;
   PDB(kDraw,1) Info("FillWeight","fWeight= %f", fWeight);
}


//______________________________________________________________________________
Bool_t TProofDraw::ProcessSingle(Long64_t entry, Int_t i)
{
   // Processes a single variable from an entry.

   Double_t w;
   Double_t v[4]; //[TTreeDrawArgsParser::fgMaxDimension];

   if (fSelect)
      w = fWeight * fSelect->EvalInstance(i);
   else
      w = fWeight;
   
   PDB(kDraw,3) Info("ProcessSingle","w[%d] = %f", i, w);

   if (w != 0.0) {
      R__ASSERT(fDimension <= TTreeDrawArgsParser::GetMaxDimension());
      for (int j = 0; j < fDimension; j++)
         v[j] = fVar[j]->EvalInstance(i);
      if (fDimension >= 1)
         PDB(kDraw,4) Info("Process","v[0] = %f", v[0]);
      DoFill(entry, w, v);
   }
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TProofDraw::Process(Long64_t entry)
{
   // Executed for each entry.

   PDB(kDraw,3) Info("Process", "enter entry = %lld", entry);

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
   for (Int_t i = 0; i < 4; i++)
      SafeDelete(fVar[i]);
   SafeDelete(fSelect);
   fManager = 0;  // This is intentional. The manager is deleted when the last formula it manages
                  // is deleted. This is unusual but was usefull for backward compatibility.
   fMultiplicity = 0;
}

//______________________________________________________________________________
void TProofDraw::SetCanvas(const char *objname)
{
   // Move to a canvas named <name>_canvas; create the canvas if not existing.
   // Used to avoid screwing up existing plots when non default names are used
   // for the final objects

   TString name = objname;
   if (!gPad) {
      gROOT->MakeDefCanvas();
      gPad->SetName(name);
      PDB(kDraw,2) Info("SetCanvas", "created canvas %s", name.Data());
   } else {
      PDB(kDraw,2)
         Info("SetCanvas", "using canvas %s", gPad->GetName());
   }
}

//______________________________________________________________________________
void TProofDraw::SetDrawAtt(TObject *o)
{
   // Set the drawing attributes from the input list

   Int_t att = -1;
   PDB(kDraw,2) Info("SetDrawAtt", "setting attributes for %s", o->GetName());

   // Line Attributes
   TAttLine *al = dynamic_cast<TAttLine *> (o);
   if (al) {
      // Line color
      if (TProof::GetParameter(fInput, "PROOF_LineColor", att) == 0)
         al->SetLineColor((Color_t)att);
      // Line style
      if (TProof::GetParameter(fInput, "PROOF_LineStyle", att) == 0)
         al->SetLineStyle((Style_t)att);
      // Line color
      if (TProof::GetParameter(fInput, "PROOF_LineWidth", att) == 0)
         al->SetLineWidth((Width_t)att);
      PDB(kDraw,2) Info("SetDrawAtt", "line:   c:%d, s:%d, wd:%d",
                                      al->GetLineColor(), al->GetLineStyle(), al->GetLineWidth());
   }

   // Marker Attributes
   TAttMarker *am = dynamic_cast<TAttMarker *> (o);
   if (am) {
      // Marker color
      if (TProof::GetParameter(fInput, "PROOF_MarkerColor", att) == 0)
         am->SetMarkerColor((Color_t)att);
      // Marker size
      if (TProof::GetParameter(fInput, "PROOF_MarkerSize", att) == 0) {
         Info("SetDrawAtt", "att: %d", att);
         Float_t msz = (Float_t)att / 1000.;
         am->SetMarkerSize((Size_t)msz);
      }
      // Marker style
      if (TProof::GetParameter(fInput, "PROOF_MarkerStyle", att) == 0)
         am->SetMarkerStyle((Style_t)att);
      PDB(kDraw,2) Info("SetDrawAtt", "marker: c:%d, s:%d, sz:%f",
                                      am->GetMarkerColor(), am->GetMarkerStyle(), am->GetMarkerSize());
   }

   // Area Fill Attributes
   TAttFill *af = dynamic_cast<TAttFill *> (o);
   if (af) {
      // Area fill color
      if (TProof::GetParameter(fInput, "PROOF_FillColor", att) == 0)
         af->SetFillColor((Color_t)att);
      // Area fill style
      if (TProof::GetParameter(fInput, "PROOF_FillStyle", att) == 0)
         af->SetFillStyle((Style_t)att);
      PDB(kDraw,2) Info("SetDrawAtt", "area:   c:%d, s:%d",
                                      af->GetFillColor(), af->GetFillStyle());
   }
}

//______________________________________________________________________________
void TProofDraw::SetError(const char *sub, const char *mesg)
{
   // Sets the error status.

   if (fStatus == 0) {
      if (!(fStatus = dynamic_cast<TStatus*>(fOutput->FindObject("PROOF_Status"))))
         return;
   }

   TString m;
   m.Form("%s::%s: %s", IsA()->GetName(), sub, mesg);
   fStatus->Add(m);
}


//______________________________________________________________________________
Bool_t TProofDraw::CompileVariables()
{
   // Compiles each variable from fTreeDrawArgsParser for the tree fTree.
   // Return kFALSE if any of the variable is not compilable.

   fDimension = fTreeDrawArgsParser.GetDimension();
   fMultiplicity = 0;
   fObjEval = kFALSE;
   if (strlen(fTreeDrawArgsParser.GetSelection())) {
      fSelect = new TTreeFormula("Selection", fTreeDrawArgsParser.GetSelection(), fTree);
      fSelect->SetQuickLoad(kTRUE);
      if (!fSelect->GetNdim()) {delete fSelect; fSelect = 0; return kFALSE; }
   }

   fManager = new TTreeFormulaManager();
   if (fSelect) fManager->Add(fSelect);
   fTree->ResetBit(TTree::kForceRead);

   for (int i = 0; i < fDimension; i++) {
      fVar[i] = new TTreeFormula(Form("Var%d", i),fTreeDrawArgsParser.GetVarExp(i),fTree);
      fVar[i]->SetQuickLoad(kTRUE);
      if (!fVar[i]->GetNdim()) {
         ClearFormula();
         Error("CompileVariables", "Error compiling expression");
         SetError("CompileVariables", "Error compiling variables");

         return kFALSE;
      }
      fManager->Add(fVar[i]);
   }
   
   fManager->Sync();
   if (fManager->GetMultiplicity()==-1) fTree->SetBit(TTree::kForceRead);
   if (fManager->GetMultiplicity()>=1) fMultiplicity = fManager->GetMultiplicity();

   return kTRUE;
#if 0
   // Commenting out to silence Coverity:
   // but why was this made inactive? 
   if (fDimension==1) {
      TClass *cl = fVar[0]->EvalClass();
      if (cl) {
         fObjEval = kTRUE;
      }
   }
   return kTRUE;
#endif
}


ClassImp(TProofDrawHist)


//______________________________________________________________________________
void TProofDrawHist::Begin1D(TTree *)
{
   // Initialization for 1D Histogram.

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 1);
   TObject* orig = fTreeDrawArgsParser.GetOriginal();
   TH1* hold;
   if (fTreeDrawArgsParser.GetNoParameters() == 0 && (hold = dynamic_cast<TH1*> (orig))) {
      TH1* hnew = (TH1*) hold->Clone();
      hnew->Reset();
      fInput->Add(hnew);
   } else {
      delete orig;
      DefVar1D();
   }
}


//______________________________________________________________________________
void TProofDrawHist::Begin2D(TTree *)
{
   // Initialization for 2D histogram.

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 2);
   TObject* orig = fTreeDrawArgsParser.GetOriginal();
   TH2* hold;
   if (fTreeDrawArgsParser.GetNoParameters() == 0 && (hold = dynamic_cast<TH2*> (orig))) {
      TH2* hnew = (TH2*) hold->Clone();
      hnew->Reset();
      fInput->Add(hnew);
   } else {
      delete orig;
      DefVar2D();
   }
}


//______________________________________________________________________________
void TProofDrawHist::Begin3D(TTree *)
{
   // Initialization for 3D histogram.

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 3);
   TObject* orig = fTreeDrawArgsParser.GetOriginal();
   TH3* hold;
   if ((hold = dynamic_cast<TH3*> (orig)) && fTreeDrawArgsParser.GetNoParameters() == 0) {
      TH3* hnew = (TH3*) hold->Clone();
      hnew->Reset();
      fInput->Add(hnew);
   } else {
      delete orig;
      DefVar3D();
   }
}

//______________________________________________________________________________
void TProofDrawHist::Begin(TTree *tree)
{
   // See TProofDraw::Begin().

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   if (fTreeDrawArgsParser.GetObjectName() == "")
      fTreeDrawArgsParser.SetObjectName("htemp");

   switch (fTreeDrawArgsParser.GetDimension()) {
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
void TProofDrawHist::DefVar1D()
{
   // Define vars for 1D Histogram.

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 1);

   fTreeDrawArgsParser.SetOriginal(0);
   TString exp = fTreeDrawArgsParser.GetVarExp();
   exp += ">>";
   double binsx, minx, maxx;
   if (fTreeDrawArgsParser.IsSpecified(0))
      gEnv->SetValue("Hist.Binning.1D.x", fTreeDrawArgsParser.GetParameter(0));
   binsx = gEnv->GetValue("Hist.Binning.1D.x",100);
   minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
   maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
   exp += fTreeDrawArgsParser.GetObjectName();
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
      Error("DefVar1D", "Cannot find varexp on the fInput");
   if (fTreeDrawArgsParser.GetNoParameters() != 3)
      fInput->Add(new TNamed("PROOF_OPTIONS", "rebin"));
}

//______________________________________________________________________________
void TProofDrawHist::DefVar2D()
{
   // Define variables for 2D histogram.

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 2);

   fTreeDrawArgsParser.SetOriginal(0);
   TString exp = fTreeDrawArgsParser.GetVarExp();
   exp += ">>";
   double binsx, minx, maxx;
   double binsy, miny, maxy;
   if (fTreeDrawArgsParser.IsSpecified(0))
      gEnv->SetValue("Hist.Binning.2D.x", fTreeDrawArgsParser.GetParameter(0));
   if (fTreeDrawArgsParser.IsSpecified(3))
      gEnv->SetValue("Hist.Binning.2D.y", fTreeDrawArgsParser.GetParameter(3));
   binsx = gEnv->GetValue("Hist.Binning.2D.x",100);
   minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
   maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
   binsy = gEnv->GetValue("Hist.Binning.2D.y",100);
   miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
   maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
   exp += fTreeDrawArgsParser.GetObjectName();
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
      Error("DefVar2D", "Cannot find varexp on the fInput");
   if (fTreeDrawArgsParser.GetNoParameters() != 6)
      fInput->Add(new TNamed("PROOF_OPTIONS", "rebin"));
}

//______________________________________________________________________________
void TProofDrawHist::DefVar3D()
{
   // Define variables for 3D histogram.

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 3);

   fTreeDrawArgsParser.SetOriginal(0);
   TString exp = fTreeDrawArgsParser.GetVarExp();
   exp += ">>";
   double binsx, minx, maxx;
   double binsy, miny, maxy;
   double binsz, minz, maxz;
   if (fTreeDrawArgsParser.IsSpecified(0))
      gEnv->SetValue("Hist.Binning.3D.x", fTreeDrawArgsParser.GetParameter(0));
   if (fTreeDrawArgsParser.IsSpecified(3))
      gEnv->SetValue("Hist.Binning.3D.y", fTreeDrawArgsParser.GetParameter(3));
   if (fTreeDrawArgsParser.IsSpecified(6))
      gEnv->SetValue("Hist.Binning.3D.z", fTreeDrawArgsParser.GetParameter(6));
   binsx = gEnv->GetValue("Hist.Binning.3D.x",100);
   minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
   maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
   binsy = gEnv->GetValue("Hist.Binning.3D.y",100);
   miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
   maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
   binsz = gEnv->GetValue("Hist.Binning.3D.z",100);
   minz =  fTreeDrawArgsParser.GetIfSpecified(7, 0);
   maxz =  fTreeDrawArgsParser.GetIfSpecified(8, 0);
   exp += fTreeDrawArgsParser.GetObjectName();
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
      Error("DefVar3D", "Cannot find varexp on the fInput");
   if (fTreeDrawArgsParser.GetNoParameters() != 9)
      fInput->Add(new TNamed("PROOF_OPTIONS", "rebin"));
}

//______________________________________________________________________________
void TProofDrawHist::DefVar()
{
   // Define variables according to arguments.

   PDB(kDraw,1) Info("DefVar","Enter");

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   if (fTreeDrawArgsParser.GetObjectName() == "")
      fTreeDrawArgsParser.SetObjectName("htemp");

   switch (fTreeDrawArgsParser.GetDimension()) {
      case 1:
         DefVar1D();
         break;
      case 2:
         DefVar2D();
         break;
      case 3:
         DefVar3D();
         break;
      default:
         Error("DefVar", "Wrong dimension");
         break;
   }
   PDB(kDraw,1) Info("DefVar","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("DefVar","varexp: %s", fInitialExp.Data());
   fTree = 0;
}

//______________________________________________________________________________
void TProofDrawHist::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);
   if (fTree == 0) {
      if (!dynamic_cast<TH1*> (fTreeDrawArgsParser.GetOriginal())) {
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

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   SafeDelete(fHistogram);

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   fDimension = fTreeDrawArgsParser.GetDimension();
   TString exp = fTreeDrawArgsParser.GetExp();
   const char *objname = fTreeDrawArgsParser.GetObjectName();
   if (objname && strlen(objname) > 0 && strcmp(objname, "htemp")) {
      TH1 *hist = dynamic_cast<TH1*> (fInput->FindObject(objname));
      if (hist) {
         fHistogram = (TH1 *) hist->Clone();
         PDB(kDraw,1) Info("SlaveBegin","original histogram found");
      } else {
         PDB(kDraw,1) Info("SlaveBegin", "original object '%s' not found"
                                         " or it is not a histogram", objname);
      }
   }

   // Create the histogram if not found in the input list
   if (!fHistogram) {
      Int_t countx = 100; double minx = 0, maxx = 0;
      Int_t county = 100; double miny = 0, maxy = 0;
      Int_t countz = 100; double minz = 0, maxz = 0;
      if (fTreeDrawArgsParser.GetNoParameters() != 0) {
         countx = (Int_t) fTreeDrawArgsParser.GetIfSpecified(0, countx);
         county = (Int_t) fTreeDrawArgsParser.GetIfSpecified(3, county);
         countz = (Int_t) fTreeDrawArgsParser.GetIfSpecified(6, countz);
         minx =  fTreeDrawArgsParser.GetIfSpecified(1, minx);
         maxx =  fTreeDrawArgsParser.GetIfSpecified(2, maxx);
         miny =  fTreeDrawArgsParser.GetIfSpecified(4, miny);
         maxy =  fTreeDrawArgsParser.GetIfSpecified(5, maxy);
         minz =  fTreeDrawArgsParser.GetIfSpecified(7, minz);
         maxz =  fTreeDrawArgsParser.GetIfSpecified(8, maxz);
      }
      if (fTreeDrawArgsParser.GetNoParameters() != 3*fDimension)
         Error("SlaveBegin", "Impossible - Wrong number of parameters");

      if (fDimension == 1)
         fHistogram = new TH1F(fTreeDrawArgsParser.GetObjectName(),
                              fTreeDrawArgsParser.GetObjectTitle(),
                              countx, minx, maxx);
      else if (fDimension == 2){
         fHistogram = new TH2F(fTreeDrawArgsParser.GetObjectName(),
                              fTreeDrawArgsParser.GetObjectTitle(),
                              countx, minx, maxx,
                              county, miny, maxy);
      }
      else if (fDimension == 3) {
         fHistogram = new TH3F(fTreeDrawArgsParser.GetObjectName(),
                              fTreeDrawArgsParser.GetObjectTitle(),
                              countx, minx, maxx,
                              county, miny, maxy,
                              countz, minz, maxz);
      } else {
         Info("Begin", "Wrong dimension");
         return;        // FIXME: end the session
      }
      if (minx >= maxx)
         fHistogram->SetBuffer(TH1::GetDefaultBufferSize());
      if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("PROOF_OPTIONS"))) {
         if (strstr(opt->GetTitle(), "rebin"))
            fHistogram->SetBit(TH1::kCanRebin);
      }
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

   fHistogram = (TH1F *) fOutput->FindObject(fTreeDrawArgsParser.GetObjectName());
   if (fHistogram) {
      SetStatus((Int_t) fHistogram->GetEntries());
      TH1 *h = 0;
      if ((h = dynamic_cast<TH1*> (fTreeDrawArgsParser.GetOriginal()))) {
         if (!fTreeDrawArgsParser.GetAdd())
            h->Reset();
         TList l;
         l.Add(fHistogram);
         h->Merge(&l);
         fOutput->Remove(fHistogram);
         delete fHistogram;
      } else {
         // Set the title
         fHistogram->SetTitle(fTreeDrawArgsParser.GetObjectTitle());
         h = fHistogram;
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         // Choose the right canvas
         SetCanvas(h->GetName());
         // Draw
         SetDrawAtt(h);
         h->Draw(fOption.Data());
      }
   }
   fHistogram = 0;
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

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   SafeDelete(fEventLists);

   fDimension = 0;
   fTree = 0;
   fEventLists = new TList();
   fEventLists->SetName("PROOF_EventListsList");
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

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   TEventList *el = dynamic_cast<TEventList*> (fOutput->FindObject("PROOF_EventList"));
   if (el) {
      el->SetName(fInitialExp.Data()+2);
      SetStatus(el->GetN());
      if (TEventList* old = dynamic_cast<TEventList*> (fTreeDrawArgsParser.GetOriginal())) {
         if (!fTreeDrawArgsParser.GetAdd())
            old->Reset();
         old->Add(el);
         fOutput->Remove(el);
         delete el;
      }
   }
   else
      Error("Terminate", "Cannot find output EventList");

}

ClassImp(TProofDrawEntryList)

//______________________________________________________________________________
TProofDrawEntryList::~TProofDrawEntryList()
{
   //class destructor
   SafeDelete(fElist);
}

//______________________________________________________________________________
void TProofDrawEntryList::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);

   fTree = tree;
   CompileVariables();
}

//______________________________________________________________________________
void TProofDrawEntryList::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   SafeDelete(fElist);

   fDimension = 0;
   fTree = 0;
   fElist = new TEntryList("PROOF_EntryList", "PROOF_EntryList");
   fOutput->Add(fElist);

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}

//______________________________________________________________________________
void TProofDrawEntryList::DoFill(Long64_t entry, Double_t , const Double_t *)
{
   // Fills the eventlist with given values.

   fElist->Enter(entry);
}

//______________________________________________________________________________
void TProofDrawEntryList::SlaveTerminate(void)
{
   // See TProofDraw::SlaveTerminate().

   PDB(kDraw,1) Info("SlaveTerminate","Enter");
   fElist->OptimizeStorage();
   fElist = 0;
}

//______________________________________________________________________________
void TProofDrawEntryList::Terminate(void)
{
   // See TProofDraw::Terminate().

   TProofDraw::Terminate();   // take care of fStatus
   if (!fStatus)
      return;

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   TEntryList *el = dynamic_cast<TEntryList*> (fOutput->FindObject("PROOF_EntryList"));

   if (el) {
      el->SetName(fInitialExp.Data()+2);
      SetStatus(el->GetN());
      if (TEntryList* old = dynamic_cast<TEntryList*> (fTreeDrawArgsParser.GetOriginal())) {
         if (!fTreeDrawArgsParser.GetAdd())
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
      if (!dynamic_cast<TProfile*> (fTreeDrawArgsParser.GetOriginal())) {
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
void TProofDrawProfile::DefVar()
{
   // Define relevant variables

   PDB(kDraw,1) Info("DefVar","Enter");

   if (fTreeDrawArgsParser.GetDimension() < 0) {

      // Init parser
      fSelection = fInput->FindObject("selection")->GetTitle();
      fInitialExp = fInput->FindObject("varexp")->GetTitle();

      fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   }

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 2);

   fTreeDrawArgsParser.SetOriginal(0);
   TString exp = fTreeDrawArgsParser.GetVarExp();
   exp += ">>";
   double binsx, minx, maxx;
   if (fTreeDrawArgsParser.IsSpecified(0))
      gEnv->SetValue("Hist.Binning.2D.Prof", fTreeDrawArgsParser.GetParameter(0));
   binsx = gEnv->GetValue("Hist.Binning.2D.Prof",100);
   minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
   maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
   if (fTreeDrawArgsParser.GetObjectName() == "")
      fTreeDrawArgsParser.SetObjectName("htemp");
   exp += fTreeDrawArgsParser.GetObjectName();
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
      Error("DefVar", "Cannot find varexp on the fInput");
   if (fTreeDrawArgsParser.GetNoParameters() != 3)
      fInput->Add(new TNamed("PROOF_OPTIONS", "rebin"));
}

//______________________________________________________________________________
void TProofDrawProfile::Begin(TTree *tree)
{
   // See TProofDraw::Begin().

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 2);

   TObject *orig = fTreeDrawArgsParser.GetOriginal();
   TH1* pold;
   if ((pold = dynamic_cast<TProfile*> (orig)) && fTreeDrawArgsParser.GetNoParameters() == 0) {
      TProfile* pnew = (TProfile*) pold->Clone();
      pnew->Reset();
      fInput->Add(pnew);
   } else {
      delete orig;
      DefVar();
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

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   SafeDelete(fProfile);


   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   fDimension = 2;
   TString exp = fTreeDrawArgsParser.GetExp();

   if (fTreeDrawArgsParser.GetOriginal()) {
      fProfile = dynamic_cast<TProfile*> (fTreeDrawArgsParser.GetOriginal());
      if (fProfile) {
         fOutput->Add(fProfile);
         PDB(kDraw,1) Info("SlaveBegin","Original profile histogram found");
         return;
      }
      else
         Error("SlaveBegin","Original object found but it is not a histogram");
   }
   Int_t countx = 100; double minx = 0, maxx = 0;
   if (fTreeDrawArgsParser.GetNoParameters() != 0) {
      countx = (Int_t) fTreeDrawArgsParser.GetIfSpecified(0, countx);
      minx =  fTreeDrawArgsParser.GetIfSpecified(1, minx);
      maxx =  fTreeDrawArgsParser.GetIfSpecified(2, maxx);
   }
   if (fTreeDrawArgsParser.GetNoParameters() != 3)
      Error("SlaveBegin", "Impossible - Wrong number of parameters");
   TString constructorOptions = "";
   if (fOption.Contains("profs"))
      constructorOptions = "s";
   else if (fOption.Contains("profi"))
      constructorOptions = "i";
   else if (fOption.Contains("profg"))
      constructorOptions = "g";

   fProfile = new TProfile(fTreeDrawArgsParser.GetObjectName(),
                           fTreeDrawArgsParser.GetObjectTitle(),
                           countx, minx, maxx,
                           constructorOptions);
   if (minx >= maxx)
      fProfile->SetBuffer(TH1::GetDefaultBufferSize());

   if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("PROOF_OPTIONS"))) {
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

   fProfile = (TProfile *) fOutput->FindObject(fTreeDrawArgsParser.GetObjectName());
   if (fProfile) {
      SetStatus((Int_t) fProfile->GetEntries());
      TProfile *pf = 0;
      if ((pf = dynamic_cast<TProfile*> (fTreeDrawArgsParser.GetOriginal()))) {
         if (!fTreeDrawArgsParser.GetAdd())
            pf->Reset();
         TList l;
         l.Add(fProfile);
         pf->Merge(&l);
         fOutput->Remove(fProfile);
         delete fProfile;
      } else {
         fProfile->SetTitle(fTreeDrawArgsParser.GetObjectTitle());
         pf = fProfile;
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         // Choose the right canvas
         SetCanvas(pf->GetName());
         // Draw
         SetDrawAtt(pf);
         pf->Draw(fOption.Data());
      }
   }
   fProfile = 0;
}


ClassImp(TProofDrawProfile2D)

//______________________________________________________________________________
void TProofDrawProfile2D::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);
   if (fTree == 0) {
      if (!dynamic_cast<TProfile2D*> (fTreeDrawArgsParser.GetOriginal())) {
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
void TProofDrawProfile2D::DefVar()
{
   // Define relevant variables

   PDB(kDraw,1) Info("DefVar","Enter");

   if (fTreeDrawArgsParser.GetDimension() < 0) {

      // Init parser
      fSelection = fInput->FindObject("selection")->GetTitle();
      fInitialExp = fInput->FindObject("varexp")->GetTitle();

      fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   }
   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 3);

   fTreeDrawArgsParser.SetOriginal(0);
   TString exp = fTreeDrawArgsParser.GetVarExp();
   exp += ">>";
   double binsx, minx, maxx;
   double binsy, miny, maxy;
   if (fTreeDrawArgsParser.IsSpecified(0))
      gEnv->SetValue("Hist.Binning.3D.Profx", fTreeDrawArgsParser.GetParameter(0));
   if (fTreeDrawArgsParser.IsSpecified(3))
      gEnv->SetValue("Hist.Binning.3D.Profy", fTreeDrawArgsParser.GetParameter(3));
   binsx = gEnv->GetValue("Hist.Binning.3D.Profx",20);
   minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
   maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
   binsy = gEnv->GetValue("Hist.Binning.3D.Profy",20);
   miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
   maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
   if (fTreeDrawArgsParser.GetObjectName() == "")
      fTreeDrawArgsParser.SetObjectName("htemp");
   exp += fTreeDrawArgsParser.GetObjectName();
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
      Error("DefVar", "Cannot find varexp on the fInput");
   if (fTreeDrawArgsParser.GetNoParameters() != 6)
      fInput->Add(new TNamed("PROOF_OPTIONS", "rebin"));
}

//______________________________________________________________________________
void TProofDrawProfile2D::Begin(TTree *tree)
{
   // See TProofDraw::Begin().

   PDB(kDraw,1) Info("Begin","Enter tree = %p", tree);

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 3);

   TObject *orig = fTreeDrawArgsParser.GetOriginal();
   TProfile2D *pold;
   if ((pold = dynamic_cast<TProfile2D*> (orig)) && fTreeDrawArgsParser.GetNoParameters() == 0) {
      TProfile2D* pnew = (TProfile2D*) pold->Clone();
      pnew->Reset();
      fInput->Add(pnew);
   } else {
      delete orig;
      DefVar();
   }

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}

//______________________________________________________________________________
void TProofDrawProfile2D::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();

   SafeDelete(fProfile);

   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   fDimension = 2;
   TString exp = fTreeDrawArgsParser.GetExp();

   if (fTreeDrawArgsParser.GetOriginal()) {
      fProfile = dynamic_cast<TProfile2D*> (fTreeDrawArgsParser.GetOriginal());
      if (fProfile) {
         fOutput->Add(fProfile);
         PDB(kDraw,1) Info("SlaveBegin","Original profile histogram found");
         return;
      } else
         Error("SlaveBegin","Original object found but it is not a histogram");
   }
   Int_t countx = 40; double minx = 0, maxx = 0;
   Int_t county = 40; double miny = 0, maxy = 0;
   if (fTreeDrawArgsParser.GetNoParameters() != 0) {
      countx = (Int_t) fTreeDrawArgsParser.GetIfSpecified(0, countx);
      minx =  fTreeDrawArgsParser.GetIfSpecified(1, minx);
      maxx =  fTreeDrawArgsParser.GetIfSpecified(2, maxx);
      county = (Int_t) fTreeDrawArgsParser.GetIfSpecified(3, countx);
      miny =  fTreeDrawArgsParser.GetIfSpecified(4, minx);
      maxy =  fTreeDrawArgsParser.GetIfSpecified(5, maxx);
   }
   if (fTreeDrawArgsParser.GetNoParameters() != 6)
      Error("SlaveBegin", "Impossible - Wrong number of parameters");

   TString constructorOptions = "";
   if (fOption.Contains("profs"))
      constructorOptions = "s";
   else if (fOption.Contains("profi"))
      constructorOptions = "i";
   else if (fOption.Contains("profg"))
      constructorOptions = "g";

   fProfile = new TProfile2D(fTreeDrawArgsParser.GetObjectName(),
                             fTreeDrawArgsParser.GetObjectTitle(),
                             countx, minx, maxx,
                             county, miny, maxy,
                             constructorOptions);
   if (minx >= maxx)
      fProfile->SetBuffer(TH1::GetDefaultBufferSize());

   if (TNamed *opt = dynamic_cast<TNamed*> (fInput->FindObject("PROOF_OPTIONS"))) {
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

   fProfile = (TProfile2D *) fOutput->FindObject(fTreeDrawArgsParser.GetObjectName());
   if (fProfile) {
      SetStatus((Int_t) fProfile->GetEntries());
      TProfile2D *pf = 0;
      if ((pf = dynamic_cast<TProfile2D*> (fTreeDrawArgsParser.GetOriginal()))) {
         if (!fTreeDrawArgsParser.GetAdd())
            pf->Reset();
         TList l;
         l.Add(fProfile);
         pf->Merge(&l);
         fOutput->Remove(fProfile);
         delete fProfile;
      } else {
         fProfile->SetTitle(fTreeDrawArgsParser.GetObjectTitle());
         pf = fProfile;
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         // Choose the right canvas
         SetCanvas(pf->GetName());
         // Draw
         SetDrawAtt(pf);
         pf->Draw(fOption.Data());
      }
   }
   fProfile = 0;
}


ClassImp(TProofDrawGraph)

//______________________________________________________________________________
void TProofDrawGraph::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);

   if (fTree == 0) {
      R__ASSERT(fGraph);
      fGraph->SetMarkerStyle(tree->GetMarkerStyle());
      fGraph->SetMarkerColor(tree->GetMarkerColor());
      fGraph->SetMarkerSize(tree->GetMarkerSize());
      fGraph->SetLineColor(tree->GetLineColor());
      fGraph->SetLineStyle(tree->GetLineStyle());
      fGraph->SetFillColor(tree->GetFillColor());
      fGraph->SetFillStyle(tree->GetFillStyle());
   }
   fTree = tree;
   CompileVariables();
}


//______________________________________________________________________________
void TProofDrawGraph::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);

   SafeDelete(fGraph);
   fDimension = 2;

   fGraph = new TGraph();
   fGraph->SetName("PROOF_GRAPH");
   fOutput->Add(fGraph);                         // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawGraph::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the graph with the given values.

   fGraph->SetPoint(fGraph->GetN(), v[1], v[0]);
}


//______________________________________________________________________________
void TProofDrawGraph::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fGraph = dynamic_cast<TGraph*> (fOutput->FindObject("PROOF_GRAPH"));
   if (fGraph) {
      SetStatus((Int_t) fGraph->GetN());
      TH2F* hist;
      TObject *orig = fTreeDrawArgsParser.GetOriginal();
      if ( (hist = dynamic_cast<TH2F*> (orig)) == 0 ) {
         delete orig;
         fTreeDrawArgsParser.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         if (fTreeDrawArgsParser.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.2D.x", fTreeDrawArgsParser.GetParameter(0));
         if (fTreeDrawArgsParser.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.2D.y", fTreeDrawArgsParser.GetParameter(3));
         binsx = gEnv->GetValue("Hist.Binning.2D.x",100);
         minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
         maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.2D.y",100);
         miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
         maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
         hist = new TH2F(fTreeDrawArgsParser.GetObjectName(), fTreeDrawArgsParser.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx, (Int_t) binsy, miny, maxy);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fTreeDrawArgsParser.GetNoParameters() != 6)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);
//         if (fTreeDrawArgsParser.GetShouldDraw())    // ?? FIXME
//            hist->SetDirectory(0);
      } else {
         if (!fTreeDrawArgsParser.GetAdd())
            hist->Reset();
      }
      if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
         Double_t* xArray = fGraph->GetX();
         Double_t* yArray = fGraph->GetY();
         Double_t xmin = *std::min_element(xArray, xArray+fGraph->GetN());
         Double_t xmax = *std::max_element(xArray, xArray+fGraph->GetN());
         Double_t ymin = *std::min_element(yArray, yArray+fGraph->GetN());
         Double_t ymax = *std::max_element(yArray, yArray+fGraph->GetN());
         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,xmin,xmax,ymin,ymax);
      }
      if (!hist->TestBit(kCanDelete)) {
         TH1 *h2c = hist->DrawCopy(fOption.Data());
         h2c->SetStats(kFALSE);
      } else {
         SetDrawAtt(hist);
         hist->Draw();
      }
      gPad->Update();

      fGraph->SetEditable(kFALSE);
      // FIXME set color, marker size, etc.

      if (fTreeDrawArgsParser.GetShouldDraw()) {
         SetDrawAtt(fGraph);
         if (fOption == "" || strcmp(fOption, "same") == 0)
            fGraph->Draw("p");
         else
            fGraph->Draw(fOption);
         gPad->Update();
      }
      if (!hist->TestBit(kCanDelete)) {
         for (int i = 0; i < fGraph->GetN(); i++) {
            Double_t x, y;
            fGraph->GetPoint(i, x, y);
            hist->Fill(x, y, 1);
         }
      }
   }
   fGraph = 0;
}


ClassImp(TProofDrawPolyMarker3D)

//______________________________________________________________________________
void TProofDrawPolyMarker3D::Init(TTree *tree)
{
   // See TProofDraw::Init().

   PDB(kDraw,1) Info("Init","Enter tree = %p", tree);

   if (fTree == 0) {
      R__ASSERT(fPolyMarker3D);
      fPolyMarker3D->SetMarkerStyle(tree->GetMarkerStyle());
      fPolyMarker3D->SetMarkerColor(tree->GetMarkerColor());
      fPolyMarker3D->SetMarkerSize(tree->GetMarkerSize());
   }
   fTree = tree;
   CompileVariables();
}

//______________________________________________________________________________
void TProofDrawPolyMarker3D::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 3);

   SafeDelete(fPolyMarker3D);
   fDimension = 3;

   fPolyMarker3D = new TPolyMarker3D();
   fOutput->Add(fPolyMarker3D);      // release ownership

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawPolyMarker3D::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the scatter plot with the given values.

   fPolyMarker3D->SetNextPoint(v[2], v[1], v[0]);
}


//______________________________________________________________________________
void TProofDrawPolyMarker3D::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fPolyMarker3D = 0;
   TIter next(fOutput);
   while (TObject* o = next()) {
      if (dynamic_cast<TPolyMarker3D*> (o)) {
         fPolyMarker3D = dynamic_cast<TPolyMarker3D*> (o);
         break;
      }
   }

   Bool_t checkPrevious = kFALSE;
   if (fPolyMarker3D) {
      SetStatus((Int_t) fPolyMarker3D->Size());
      TH3F* hist;
      TObject *orig = fTreeDrawArgsParser.GetOriginal();
      if ( (hist = dynamic_cast<TH3F*> (orig)) == 0 ) {
         delete orig;
         fTreeDrawArgsParser.SetOriginal(0);
         if (fOption.Contains("same")) {
            // Check existing histogram
            hist = dynamic_cast<TH3F *> (gDirectory->Get(fTreeDrawArgsParser.GetObjectName()));
         }
         if (!hist) {
            double binsx, minx, maxx;
            double binsy, miny, maxy;
            double binsz, minz, maxz;
            if (fTreeDrawArgsParser.IsSpecified(0))
               gEnv->SetValue("Hist.Binning.3D.x", fTreeDrawArgsParser.GetParameter(0));
            if (fTreeDrawArgsParser.IsSpecified(3))
               gEnv->SetValue("Hist.Binning.3D.y", fTreeDrawArgsParser.GetParameter(3));
            if (fTreeDrawArgsParser.IsSpecified(6))
               gEnv->SetValue("Hist.Binning.3D.z", fTreeDrawArgsParser.GetParameter(6));
            binsx = gEnv->GetValue("Hist.Binning.3D.x",100);
            minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
            maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
            binsy = gEnv->GetValue("Hist.Binning.3D.y",100);
            miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
            maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
            binsz = gEnv->GetValue("Hist.Binning.3D.z",100);
            minz =  fTreeDrawArgsParser.GetIfSpecified(7, 0);
            maxz =  fTreeDrawArgsParser.GetIfSpecified(8, 0);
            hist = new TH3F(fTreeDrawArgsParser.GetObjectName(), fTreeDrawArgsParser.GetObjectTitle(),
                           (Int_t) binsx, minx, maxx,
                           (Int_t) binsy, miny, maxy,
                           (Int_t) binsz, minz, maxz);
            hist->SetBit(TH1::kNoStats);
            hist->SetBit(kCanDelete);
            if (fTreeDrawArgsParser.GetNoParameters() != 9)
               hist->SetBit(TH1::kCanRebin);
            else
               hist->ResetBit(TH1::kCanRebin);
         } else {
            checkPrevious = kTRUE;
            PDB(kDraw,2)
               Info("Terminate", "found histo '%s' in gDirectory",
                                 fTreeDrawArgsParser.GetObjectName().Data());
         }
      } else {
         if (!fTreeDrawArgsParser.GetAdd())
            hist->Reset();
      }

      // Set the ranges; take into account previous histos for 'same' runs
      Double_t rmin[3], rmax[3];
      if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
         rmin[0] = rmax[0] = rmin[1] = rmax[1] = rmin[2] = rmax[2] = 0;
         if (fPolyMarker3D->Size() > 0) {
            fPolyMarker3D->GetPoint(0, rmin[0], rmin[1], rmin[2]);
            fPolyMarker3D->GetPoint(0, rmax[0], rmax[1], rmax[2]);
         }
         for (int i = 1; i < fPolyMarker3D->Size(); i++) {
            Double_t v[3];
            fPolyMarker3D->GetPoint(i, v[0], v[1], v[2]);
            for (int ii = 0; ii < 3; ii++) {
               if (v[ii] < rmin[ii]) rmin[ii] = v[ii];
               if (v[ii] > rmax[ii]) rmax[ii] = v[ii];
            }
         }
         // Compare with previous histo, if any
         if (checkPrevious) {
            rmin[0] = (hist->GetXaxis()->GetXmin() < rmin[0]) ? hist->GetXaxis()->GetXmin()
                                                              : rmin[0];
            rmin[1] = (hist->GetYaxis()->GetXmin() < rmin[1]) ? hist->GetYaxis()->GetXmin()
                                                              : rmin[1];
            rmin[2] = (hist->GetZaxis()->GetXmin() < rmin[2]) ? hist->GetZaxis()->GetXmin()
                                                              : rmin[2];
            rmax[0] = (hist->GetXaxis()->GetXmax() > rmax[0]) ? hist->GetXaxis()->GetXmax()
                                                              : rmax[0];
            rmax[1] = (hist->GetYaxis()->GetXmax() > rmax[1]) ? hist->GetYaxis()->GetXmax()
                                                              : rmax[1];
            rmax[2] = (hist->GetZaxis()->GetXmax() > rmax[2]) ? hist->GetZaxis()->GetXmax()
                                                              : rmax[2];
         }

         THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,
                           rmin[0], rmax[0], rmin[1], rmax[1], rmin[2], rmax[2]);
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         if (!hist->TestBit(kCanDelete)) {
            TH1 *histcopy = hist->DrawCopy(fOption.Data());
            histcopy->SetStats(kFALSE);
         } else {
            SetDrawAtt(hist);
            hist->Draw(fOption);        // no draw options on purpose
         }
         gPad->Update();
      } else {
         gPad->Clear();
         gPad->Range(-1,-1,1,1);
         TView::CreateView(1,rmin,rmax);
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         SetDrawAtt(fPolyMarker3D);
         fPolyMarker3D->Draw(fOption);
      }
      gPad->Update();
      if (!hist->TestBit(kCanDelete)) {
         for (int i = 0; i < fPolyMarker3D->Size(); i++) {
            Float_t x, y, z;
            fPolyMarker3D->GetPoint(i, x, y, z);
            hist->Fill(x, y, z, 1);
         }
      }
   }
}


ClassImp(TProofDrawListOfGraphs)

//______________________________________________________________________________
void TProofDrawListOfGraphs::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 3);

   SafeDelete(fPoints);

   fDimension = 3;

   fPoints = new TProofVectorContainer<Point3D_t>(new std::vector<Point3D_t>);
   fPoints->SetName("PROOF_SCATTERPLOT");
   fOutput->Add(fPoints);      // release ownership (? FIXME)

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawListOfGraphs::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the scatter plot with the given values.

   fPoints->GetVector()->push_back(Point3D_t(v[2], v[1], v[0]));
}


//______________________________________________________________________________
void TProofDrawListOfGraphs::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fPoints = dynamic_cast<TProofVectorContainer<Point3D_t>*>
               (fOutput->FindObject("PROOF_SCATTERPLOT"));
   if (fPoints) {
      std::vector<Point3D_t> *points = fPoints->GetVector();
      R__ASSERT(points);
      SetStatus((Int_t) points->size());
      TH2F* hist;
      TObject *orig = fTreeDrawArgsParser.GetOriginal();
      if ( (hist = dynamic_cast<TH2F*> (orig)) == 0 ) {
         delete orig;
         fTreeDrawArgsParser.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         if (fTreeDrawArgsParser.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.2D.x", fTreeDrawArgsParser.GetParameter(0));
         if (fTreeDrawArgsParser.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.2D.y", fTreeDrawArgsParser.GetParameter(3));
         binsx = gEnv->GetValue("Hist.Binning.2D.x", 40);
         minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
         maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.2D.y", 40);
         miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
         maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
         hist = new TH2F(fTreeDrawArgsParser.GetObjectName(), fTreeDrawArgsParser.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx, (Int_t) binsy, miny, maxy);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fTreeDrawArgsParser.GetNoParameters() != 6)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);

//         if (fTreeDrawArgsParser.GetShouldDraw())         // ?? FIXME
//            hist->SetDirectory(0);
      }
      Double_t rmin[3], rmax[3];

      // FIXME take rmin and rmax from the old histogram
      rmin[0] = rmax[0] = rmin[1] = rmax[1] = rmin[2] = rmax[2] = 0;
      if (points->size() > 0) {
         rmin[0] = rmax[0] = (*points)[0].fX;
         rmin[1] = rmax[1] = (*points)[0].fY;
         rmin[2] = rmax[2] = (*points)[0].fZ;

         for (vector<Point3D_t>::const_iterator i = points->begin() + 1; i < points->end(); ++i) {
            if (rmax[0] < i->fX) rmax[0] = i->fX;
            if (rmax[1] < i->fY) rmax[1] = i->fY;
            if (rmax[2] < i->fZ) rmax[2] = i->fZ;
            if (rmin[0] > i->fX) rmin[0] = i->fX;
            if (rmin[1] > i->fY) rmin[1] = i->fY;
            if (rmin[2] > i->fZ) rmin[2] = i->fZ;
         }
         // in this case we don't care about user-specified limits
         if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
            THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,
                           rmin[1], rmax[1], rmin[2], rmax[2]);
         }
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
      for (vector<Point3D_t>::const_iterator i = points->begin();
           i < points->end(); ++i) {
         col = Int_t((ncolors-1)*((i->fX-rmin[0])/(rmax[0]-rmin[0])));
         if (col < 0) col = 0;
         if (col > ncolors-1) col = ncolors-1;
         gr = (TGraph*)grs->UncheckedAt(col);
         if (gr) gr->SetPoint(gr->GetN(), i->fY, i->fZ);
      }
      // Remove potential empty graphs
      for (col=0;col<ncolors;col++) {
         gr = (TGraph*)grs->At(col);
         if (gr && gr->GetN() <= 0) grs->Remove(gr);
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         SetDrawAtt(hist);
         hist->Draw(fOption.Data());
         gPad->Update();
      }
      fOutput->Remove(fPoints);
      SafeDelete(fPoints);
   }
}


ClassImp(TProofDrawListOfPolyMarkers3D)


//______________________________________________________________________________
void TProofDrawListOfPolyMarkers3D::SlaveBegin(TTree *tree)
{
   // See TProofDraw::SlaveBegin().

   PDB(kDraw,1) Info("SlaveBegin","Enter tree = %p", tree);

   // Get the weight
   TProofDraw::FillWeight();

   fSelection = fInput->FindObject("selection")->GetTitle();
   fInitialExp = fInput->FindObject("varexp")->GetTitle();
   fTreeDrawArgsParser.Parse(fInitialExp, fSelection, fOption);
   R__ASSERT(fTreeDrawArgsParser.GetDimension() == 4);

   SafeDelete(fPoints);

   fDimension = 4;

   fPoints = new TProofVectorContainer<Point4D_t>(new std::vector<Point4D_t>);
   fPoints->SetName("PROOF_SCATTERPLOT");
   fOutput->Add(fPoints);      // release ownership (? FIXME)

   PDB(kDraw,1) Info("Begin","selection: %s", fSelection.Data());
   PDB(kDraw,1) Info("Begin","varexp: %s", fInitialExp.Data());
}


//______________________________________________________________________________
void TProofDrawListOfPolyMarkers3D::DoFill(Long64_t , Double_t , const Double_t *v)
{
   // Fills the scatter plot with the given values.

   fPoints->GetVector()->push_back(Point4D_t(v[3], v[2], v[1], v[0]));
}



//______________________________________________________________________________
void TProofDrawListOfPolyMarkers3D::Terminate(void)
{
   // See TProofDraw::Terminate().

   PDB(kDraw,1) Info("Terminate","Enter");
   TProofDraw::Terminate();
   if (!fStatus)
      return;

   fPoints = dynamic_cast<TProofVectorContainer<Point4D_t>*>
               (fOutput->FindObject("PROOF_SCATTERPLOT"));
   if (fPoints) {
      std::vector<Point4D_t> *points = fPoints->GetVector();
      R__ASSERT(points);
      SetStatus((Int_t) points->size());
      TH3F* hist;
      TObject *orig = fTreeDrawArgsParser.GetOriginal();
      if ( (hist = dynamic_cast<TH3F*> (orig)) == 0 || fTreeDrawArgsParser.GetNoParameters() != 0) {
         delete orig;
         fTreeDrawArgsParser.SetOriginal(0);
         double binsx, minx, maxx;
         double binsy, miny, maxy;
         double binsz, minz, maxz;
         if (fTreeDrawArgsParser.IsSpecified(0))
            gEnv->SetValue("Hist.Binning.3D.x", fTreeDrawArgsParser.GetParameter(0));
         if (fTreeDrawArgsParser.IsSpecified(3))
            gEnv->SetValue("Hist.Binning.3D.y", fTreeDrawArgsParser.GetParameter(3));
         if (fTreeDrawArgsParser.IsSpecified(6))
            gEnv->SetValue("Hist.Binning.3D.z", fTreeDrawArgsParser.GetParameter(3));
         binsx = gEnv->GetValue("Hist.Binning.3D.x", 20);
         minx =  fTreeDrawArgsParser.GetIfSpecified(1, 0);
         maxx =  fTreeDrawArgsParser.GetIfSpecified(2, 0);
         binsy = gEnv->GetValue("Hist.Binning.3D.y", 20);
         miny =  fTreeDrawArgsParser.GetIfSpecified(4, 0);
         maxy =  fTreeDrawArgsParser.GetIfSpecified(5, 0);
         binsz = gEnv->GetValue("Hist.Binning.3D.z", 20);
         minz =  fTreeDrawArgsParser.GetIfSpecified(7, 0);
         maxz =  fTreeDrawArgsParser.GetIfSpecified(8, 0);
         hist = new TH3F(fTreeDrawArgsParser.GetObjectName(), fTreeDrawArgsParser.GetObjectTitle(),
                        (Int_t) binsx, minx, maxx,
                        (Int_t) binsy, miny, maxy,
                        (Int_t) binsz, minz, maxz);
         hist->SetBit(TH1::kNoStats);
         hist->SetBit(kCanDelete);
         if (fTreeDrawArgsParser.GetNoParameters() != 9)
            hist->SetBit(TH1::kCanRebin);
         else
            hist->ResetBit(TH1::kCanRebin);

//         if (fTreeDrawArgsParser.GetShouldDraw())          // ?? FIXME
//            hist->SetDirectory(0);
      }
      Double_t rmin[4], rmax[4];


      // FIXME take rmin and rmax from the old histogram
      rmin[0] = rmax[0] = rmin[1] = rmax[1] = rmin[2] = rmax[2] = 0;
      if (points->size() > 0) {
         rmin[0] = rmax[0] = (*points)[0].fX;
         rmin[1] = rmax[1] = (*points)[0].fY;
         rmin[2] = rmax[2] = (*points)[0].fZ;
         rmin[3] = rmax[3] = (*points)[0].fT;

         for (vector<Point4D_t>::const_iterator i = points->begin() + 1; i < points->end(); ++i) {
            if (rmax[0] < i->fX) rmax[0] = i->fX;
            if (rmax[1] < i->fY) rmax[1] = i->fY;
            if (rmax[2] < i->fZ) rmax[2] = i->fZ;
            if (rmax[3] < i->fT) rmax[3] = i->fT;
            if (rmin[0] > i->fX) rmin[0] = i->fX;
            if (rmin[1] > i->fY) rmin[1] = i->fY;
            if (rmin[2] > i->fZ) rmin[2] = i->fZ;
            if (rmin[3] > i->fT) rmin[3] = i->fT;
         }
         // in this case we don't care about user-specified limits
         if (hist->TestBit(TH1::kCanRebin) && hist->TestBit(kCanDelete)) {
            THLimitsFinder::GetLimitsFinder()->FindGoodLimits(hist,
                              rmin[1], rmax[1], rmin[2], rmax[2], rmin[3], rmax[3]);
         }
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
      for (vector<Point4D_t>::const_iterator i = points->begin();
            i < points->end(); ++i) {
         col = Int_t(i->fX);
         if (col < 0) col = 0;
         if (col > ncolors-1) col = ncolors-1;
         pm3d = (TPolyMarker3D*)pms->UncheckedAt(col);
         pm3d->SetPoint(pm3d->GetLastPoint()+1, i->fY, i->fZ, i->fT);
      }
      if (fTreeDrawArgsParser.GetShouldDraw()) {
         SetDrawAtt(hist);
         hist->Draw(fOption.Data());
         gPad->Update();
      }
      fOutput->Remove(fPoints);
      SafeDelete(fPoints);
   }
}
