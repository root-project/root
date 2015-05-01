#ifndef BDT__HH
#define BDT__HH
#include <iostream>
#include <iomanip>
#include <fstream>

#include "tmvaglob.h"

#include "RQ_OBJECT.h"

#include "TROOT.h"
#include "TStyle.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TFile.h"
#include "TColor.h"
#include "TPaveText.h"
#include "TObjString.h"
#include "TControlBar.h"

#include "TGWindow.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"

#include "TMVA/DecisionTree.h"
#include "TMVA/Tools.h"
#include "TXMLEngine.h"

// Uncomment this only if the link problem is solved. The include statement tends
// to use the ROOT classes rather than the local TMVA release
// #include "TMVA/DecisionTree.h"
// #include "TMVA/DecisionTreeNode.h"
namespace TMVA{

   // this macro displays a decision tree read in from the weight file

   // static const Int_t kSigColorF = TColor::GetColor( "#2244a5" );  // novel blue 
   // static const Int_t kBkgColorF = TColor::GetColor( "#dd0033" );  // novel red  
   // static const Int_t kIntColorF = TColor::GetColor( "#33aa77" );  // novel green

   static const Int_t kSigColorF = TColor::GetColor( "#0000FF" );  // Pure Signal
   static const Int_t kBkgColorF = TColor::GetColor( "#FF0000" );  // Pure Backgr.
   static const Int_t kIntColorF = TColor::GetColor( "#33aa77" );  // novel green


   static const Int_t kSigColorT = 10;
   static const Int_t kBkgColorT = 10;
   static const Int_t kIntColorT = 10;



   class StatDialogBDT {  

      RQ_OBJECT("StatDialogBDT")

   public:

      StatDialogBDT( const TGWindow* p, TString wfile = "weights/TMVAClassification_BDT.weights.txt", 
                     TString methName = "BDT", Int_t itree = 0 );
      virtual ~StatDialogBDT() {
         TMVA::DecisionTreeNode::fgIsTraining=false;
         fThis = 0;
         fMain->CloseWindow();
         fMain->Cleanup();
         if(gROOT->GetListOfCanvases()->FindObject(fCanvas))
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
      TString fMethName;

      Int_t   fColorOffset;

   public:

      // static function for external deletion
      static void Delete() { if (fThis != 0) { delete fThis; fThis = 0; } }

      // slots
      void SetItree(); //*SIGNAL*
      void Redraw(); //*SIGNAL*
      void Close(); //*SIGNAL*

   private:

      static StatDialogBDT* fThis;

   };
   // ========================================================================================

   static std::vector<TControlBar*> BDT_Global__cbar;

   // intermediate GUI
   void BDT( const TString& fin = "TMVA.root" );
   void BDT_DeleteTBar(int i);
   void BDT( Int_t itree, TString wfile = "weights/TMVAnalysis_test_BDT.weights.txt", TString methName = "BDT", Bool_t useTMVAStyle = kTRUE ) ;

}
#endif
