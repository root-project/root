#include <iostream>
#include <iomanip>

#include "tmvaglob.C"

#include "RQ_OBJECT.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TFile.h"
#include "TColor.h"
#include "TPaveText.h"

#include "TGWindow.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"

// Uncomment this only if the link problem is solved. The include statement tends
// to use the ROOT classes rather than the local TMVA release
// #include "TMVA/DecisionTree.h"
// #include "TMVA/DecisionTreeNode.h"

// this macro displays a decision tree read in from the weight file

const Int_t kSigColorF = TColor::GetColor( "#2244a5" );  // novel blue 
const Int_t kBkgColorF = TColor::GetColor( "#dd0033" );  // novel red  
const Int_t kIntColorF = TColor::GetColor( "#33aa77" );  // novel green

const Int_t kSigColorT = 10;
const Int_t kBkgColorT = 10;
const Int_t kIntColorT = 10;

enum PlotType { EffPurity = 0 };

class StatDialogBDT {  

   RQ_OBJECT("StatDialogBDT")

public:

   StatDialogBDT( const TGWindow* p, TString wfile = "weights/TMVAnalysis_BDT.weights.txt", Int_t itree = 0 );
   virtual ~StatDialogBDT() {
      fMain->CloseWindow();
      fMain->Cleanup();
      delete fCanvas; 
   }
   
   // draw method
   void DrawTree( Int_t itree );

   void RaiseDialog() { if (fMain) { fMain->RaiseWindow(); fMain->Layout(); fMain->MapWindow(); } }
   
private:
   
   TGMainFrame *fMain;
   Int_t        fItree;
   Int_t        fNtrees;
   TCanvas*     fCanvas;

   TGNumberEntry* fInput;

   TGHorizontalFrame* fButtons;
   TGTextButton* fDrawButton;
   TGTextButton* fCloseButton;

   void UpdateCanvases();

   // draw methods
   TMVA::DecisionTree* ReadTree( TString * &vars, Int_t itree );
   void                DrawNode( TMVA::DecisionTreeNode *n, 
                                 Double_t x, Double_t y, Double_t xscale,  Double_t yscale, TString* vars );
   void GetNtrees();

   TString fWfile;

public:

   // slots
   void SetItree(); //*SIGNAL*
   void Redraw(); //*SIGNAL*
   void Close(); //*SIGNAL*
};

void StatDialogBDT::SetItree() 
{
   fItree = Int_t(fInput->GetNumber());
}

void StatDialogBDT::Redraw() 
{
   UpdateCanvases();
}

void StatDialogBDT::Close() 
{
   delete this;
}

StatDialogBDT::StatDialogBDT( const TGWindow* p, TString wfile, Int_t itree )
   : fMain( 0 ),
     fItree(itree),
     fNtrees(0),
     fCanvas(0),
     fInput(0),
     fButtons(0),
     fDrawButton(0),
     fCloseButton(0),
     fWfile( wfile )
{
   UInt_t totalWidth  = 500;
   UInt_t totalHeight = 200;

   // read number of decision trees from weight file
   GetNtrees();

   // main frame
   fMain = new TGMainFrame(p, totalWidth, totalHeight, kMainFrame | kVerticalFrame);

   TGLabel *sigLab = new TGLabel( fMain, Form( "Decision tree [%i-%i]",0,fNtrees-1 ) );
   fMain->AddFrame(sigLab, new TGLayoutHints(kLHintsLeft | kLHintsTop,5,5,5,5));

   fInput = new TGNumberEntry(fMain, (Double_t) fItree,5,-1,(TGNumberFormat::EStyle) 5);
   fMain->AddFrame(fInput, new TGLayoutHints(kLHintsLeft | kLHintsTop,5,5,5,5));
   fInput->Resize(100,24);

   fButtons = new TGHorizontalFrame(fMain, totalWidth,30);

   fCloseButton = new TGTextButton(fButtons,"&Close");
   fButtons->AddFrame(fCloseButton, new TGLayoutHints(kLHintsLeft | kLHintsTop));

   fDrawButton = new TGTextButton(fButtons,"&Draw");
   fButtons->AddFrame(fDrawButton, new TGLayoutHints(kLHintsRight | kLHintsTop,15));
  
   fMain->AddFrame(fButtons,new TGLayoutHints(kLHintsLeft | kLHintsBottom,5,5,5,5));

   fMain->SetWindowName("Decision tree");
   fMain->SetWMPosition(0,0);
   fMain->MapSubwindows();
   fMain->Resize(fMain->GetDefaultSize());
   fMain->MapWindow();

   fInput->Connect("ValueSet(Long_t)","StatDialogBDT",this, "SetItree()");

   fDrawButton->Connect("Clicked()","TGNumberEntry",fInput, "ValueSet(Long_t)");
   fDrawButton->Connect("Clicked()", "StatDialogBDT", this, "Redraw()");   

   fCloseButton->Connect("Clicked()", "StatDialogBDT", this, "Close()");
}

void StatDialogBDT::UpdateCanvases() 
{
   DrawTree( fItree );
}

void StatDialogBDT::GetNtrees()
{
   ifstream fin( fWfile );
   if (!fin.good( )) { // file not found --> Error
      cout << "*** ERROR: Weight file: " << fWfile << " does not exist" << endl;
      return;
   }
   
   TString dummy = "";
   
   // read total number of trees, and check whether requested tree is in range
   Int_t nc = 0;
   while (!dummy.Contains("NTrees")) { 
      fin >> dummy; 
      nc++; 
      if (nc > 200) {
         cout << endl;
         cout << "*** Huge problem: could not locate term \"NTrees\" in BDT weight file: " 
              << fWfile << endl;
         cout << "==> panic abort (please contact the TMVA authors)" << endl;
         cout << endl;
         exit(1);
      }
   }
   fin >> dummy; 
   fNtrees = dummy.ReplaceAll("\"","").Atoi();
   cout << "--- Found " << fNtrees << " decision trees in weight file" << endl;
   
   fin.close();
}

//_______________________________________________________________________
void StatDialogBDT::DrawNode( TMVA::DecisionTreeNode *n, 
                               Double_t x, Double_t y, 
                               Double_t xscale,  Double_t yscale, TString * vars) 
{
   // recursively puts an entries in the histogram for the node and its daughters
   //
   if (n->GetLeft() != NULL){
      TLine *a1 = new TLine(x-xscale/2,y,x-xscale,y-yscale/2);
      a1->SetLineWidth(2);
      a1->Draw();
      DrawNode((TMVA::DecisionTreeNode*) n->GetLeft(), x-xscale, y-yscale, xscale/2, yscale, vars);
   }
   if (n->GetRight() != NULL){
      TLine *a1 = new TLine(x+xscale/2,y,x+xscale,y-yscale/2);
      a1->SetLineWidth(2);
      a1->Draw();
      DrawNode((TMVA::DecisionTreeNode*) n->GetRight(), x+xscale, y-yscale, xscale/2, yscale, vars  );
   }

   TPaveText *t = new TPaveText(x-xscale/2,y-yscale/2,x+xscale/2,y+yscale/2, "NDC");

   t->SetBorderSize(1);

   t->SetFillStyle(1);
   if      (n->GetNodeType() ==  1) { t->SetFillColor( kSigColorF ); t->SetTextColor( kSigColorT ); }
   else if (n->GetNodeType() == -1) { t->SetFillColor( kBkgColorF ); t->SetTextColor( kBkgColorT ); }
   else if (n->GetNodeType() ==  0) { t->SetFillColor( kIntColorF ); t->SetTextColor( kIntColorT ); }

   char buffer[25];
   sprintf( buffer, "N=%f", n->GetNEvents() );
   t->AddText(buffer);
   sprintf( buffer, "S/(S+B)=%4.3f", n->GetPurity() );
   t->AddText(buffer);

   if (n->GetNodeType() == 0){
      if (n->GetCutType()){
         t->AddText(TString(vars[n->GetSelector()]+">"+=::Form("%5.3g",n->GetCutValue())));
      }else{
         t->AddText(TString(vars[n->GetSelector()]+"<"+=::Form("%5.3g",n->GetCutValue())));
      }
   }

   if      (n->GetNodeType() ==  1) { t->SetFillColor( kSigColorF ); t->SetTextColor( kSigColorT ); }
   else if (n->GetNodeType() == -1) { t->SetFillColor( kBkgColorF ); t->SetTextColor( kBkgColorT ); }
   else if (n->GetNodeType() ==  0) { t->SetFillColor( kIntColorF ); t->SetTextColor( kIntColorT ); }

   t->Draw();

   return;
}

TMVA::DecisionTree* StatDialogBDT::ReadTree( TString* &vars, Int_t itree )
{
   cout << "--- Reading Tree " << itree << " from weight file: " << fWfile << endl;
   ifstream fin( fWfile );
   if (!fin.good( )) { // file not found --> Error
      cout << "*** ERROR: Weight file: " << fWfile << " does not exist" << endl;
      return 0;
   }
   
   TString dummy = "";
   
   if (itree >= fNtrees) {
      cout << "*** ERROR: requested decision tree: " << itree 
           << ", but number of trained trees only: " << fNtrees << endl;
      return 0;
   }

   // file header with name
   while (!dummy.Contains("#VAR")) fin >> dummy;
   fin >> dummy >> dummy >> dummy; // the rest of header line
   
   // number of variables
   Int_t nVars;
   fin >> dummy >> nVars;

   // variable mins and maxes
   vars = new TString[nVars];
   for (Int_t i = 0; i < nVars; i++) fin >> vars[i] >> dummy >> dummy >> dummy;

   char buffer[20];
   char line[256];
   sprintf(buffer,"Tree %d",itree);

   while (!dummy.Contains(buffer)) {
      fin.getline(line,256);
      dummy = TString(line);
   }
   
   TMVA::DecisionTree *d = new TMVA::DecisionTree();
   d->Read(fin);

   fin.close();
   
   return d;
}

//_______________________________________________________________________
void StatDialogBDT::DrawTree( Int_t itree )
{
   TString *vars;   
   TMVA::DecisionTree* d = ReadTree( vars, itree );
   if (d == 0) return;

   UInt_t   depth = d->GetTotalTreeDepth();
   Double_t ystep = 1.0/(depth + 1.0);

   cout << "--- Tree depth: " << depth << endl;

   TStyle* TMVAStyle   = gROOT->GetStyle("Plain"); // our style is based on Plain
   Int_t   canvasColor = TMVAStyle->GetCanvasColor(); // backup

   TString buffer = Form( " Decision Tree No.: %d ",itree );
   if (!fCanvas) fCanvas = new TCanvas("c1",buffer,200,0,1000,600); 
   else          fCanvas->Clear();
   fCanvas->Draw();   

   DrawNode( (TMVA::DecisionTreeNode*)d->GetRoot(), 0.5, 1.-0.5*ystep, 0.25, ystep ,vars);
  
   // make the legend
   Double_t yup=0.99;
   Double_t ydown=yup-ystep/2.5;
   Double_t dy= ystep/2.5 * 0.2;
 
   TPaveText *whichTree = new TPaveText(0.85,ydown,0.98,yup, "NDC");
   whichTree->SetBorderSize(1);
   whichTree->SetFillStyle(1);
   whichTree->SetFillColor( TColor::GetColor( "#ffff33" ) );
   whichTree->AddText(buffer);
   whichTree->Draw();

   TPaveText *intermediate = new TPaveText(0.02,ydown,0.15,yup, "NDC");
   intermediate->SetBorderSize(1);
   intermediate->SetFillStyle(1);
   intermediate->SetFillColor( kIntColorF );
   intermediate->AddText("Intermediate Nodes");
   intermediate->SetTextColor( kIntColorT );
   intermediate->Draw();

   ydown = ydown - ystep/2.5 -dy;
   yup   = yup - ystep/2.5 -dy;
   TPaveText *signalleaf = new TPaveText(0.02,ydown ,0.15,yup, "NDC");
   signalleaf->SetBorderSize(1);
   signalleaf->SetFillStyle(1);
   signalleaf->SetFillColor( kSigColorF );
   signalleaf->AddText("Signal Leaf Nodes");
   signalleaf->SetTextColor( kSigColorT );
   signalleaf->Draw();

   ydown = ydown - ystep/2.5 -dy;
   yup   = yup - ystep/2.5 -dy;
   TPaveText *backgroundleaf = new TPaveText(0.02,ydown,0.15,yup, "NDC");
   backgroundleaf->SetBorderSize(1);
   backgroundleaf->SetFillStyle(1);
   backgroundleaf->SetFillColor( kBkgColorF );

   backgroundleaf->AddText("Backgr. Leaf Nodes");
   backgroundleaf->SetTextColor( kBkgColorT );
   backgroundleaf->Draw();

   fCanvas->Update();
   TString fname = Form("plots/BDT_%i", itree );
   cout << "--- Creating image: " << fname << endl;
   TMVAGlob::imgconv( fCanvas, fname );   

   TMVAStyle->SetCanvasColor( canvasColor );
}   
      
// ========================================================================================

// input: - No. of tree
//        - the weight file from which the tree is read
void BDT( Int_t itree=0, TString wfile = "weights/TMVAnalysis_BDT.weights.txt", Bool_t useTMVAStyle = kTRUE ) 
{
   // quick check if weight file exist
   ifstream fin( wfile );
   if (!fin.good( )) { // file not found --> Error
      cout << "*** ERROR: Weight file: " << wfile << " does not exist" << endl;
      return;
   }

   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   StatDialogBDT* gGui = new StatDialogBDT( gClient->GetRoot(), wfile, itree );

   gGui->DrawTree( itree );

   gGui->RaiseDialog();
}

