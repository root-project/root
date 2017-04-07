#include "TMVA/mvaweights.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH2.h"


// input: - Input file (result from TMVA)
//        - use of TMVA plotting TStyle
void TMVA::mvaweights( TString fin , Bool_t useTMVAStyle )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // few modifications
   TStyle *TMVAStyle = gROOT->GetStyle("Plain"); // our style is based on Plain
   TMVAStyle->SetTitleW(0.94);
   TMVAStyle->SetTitleH(.06);

   TString varx = "var3";
   TString vary = "var4";

   // switches
   const Bool_t Save_Images     = kTRUE;

   // checks if file with name "fin" is already open, and if not opens one   
   TFile* file = TMVAGlob::OpenFile( fin );  
   if (!file) {
      cout << "Cannot open flie: " << fin << endl;
      return;
   }

   // define Canvas layout here!
   const Int_t width = 500;   // size of canvas

   // this defines how many canvases we need
   TCanvas *c = 0;

   // counter variables
   Int_t countCanvas = 0;
   
   // retrieve trees
   TTree *tree = (TTree*)file->Get( "TestTree" );

   // search for the right histograms in full list of keys
   TObjArray* branches = tree->GetListOfBranches();
   for (Int_t imva=0; imva<branches->GetEntries(); imva++) {
      TBranch* b = (TBranch*)(*branches)[imva];
      TString methodS = b->GetName();
      cout << "Use MVA output of Method " << methodS <<endl; 
      
      if (!methodS.BeginsWith("MVA_") || methodS.EndsWith("_Proba")) continue;
      if (methodS.Contains("Cuts") ) continue;
      
      methodS.Remove(0,4);
      cout << "--- Found variable: \"" << methodS << "\"" << endl;      
      
      // create new canvas
      TString cname = Form("TMVA output %s",methodS.Data());
      c = new TCanvas( Form("canvas%d", countCanvas+1), cname, 
                       countCanvas*50+200, countCanvas*20, width, width*1.0 ); 
      c->Divide( 1, 1 );
          
      // set the histogram style
      Float_t xmin = tree->GetMinimum( varx );
      Float_t xmax = tree->GetMaximum( varx );
      Float_t ymin = tree->GetMinimum( vary );
      Float_t ymax = tree->GetMaximum( vary );
      
      Int_t nbin = 100;
      TH2F* frame   = new TH2F( "frame",  "frame",  nbin, xmin, xmax, nbin, ymin, ymax );
      TH2F* frameS  = new TH2F( "DataS",  "DataS",  nbin, xmin, xmax, nbin, ymin, ymax );
      TH2F* frameB  = new TH2F( "DataB",  "DataB",  nbin, xmin, xmax, nbin, ymin, ymax );
      TH2F* frameRS = new TH2F( "DataRS", "DataRS", nbin, xmin, xmax, nbin, ymin, ymax );
      TH2F* frameRB = new TH2F( "DataRB", "DataRB", nbin, xmin, xmax, nbin, ymin, ymax );

      Int_t nbinC = 20;
      TH2F* refS = new TH2F( "RefS", "RefS", nbinC, xmin, xmax, nbinC, ymin, ymax );
      TH2F* refB = new TH2F( "RefB", "RefB", nbinC, xmin, xmax, nbinC, ymin, ymax );
      
      Float_t mvaMin = tree->GetMinimum( Form( "MVA_%s", methodS.Data() ) );
      Float_t mvaMax = tree->GetMaximum( Form( "MVA_%s", methodS.Data() ) );

      // project trees
      TString expr = Form( "((MVA_%s-(%f))/(%f-(%f)))", methodS.Data(), mvaMin, mvaMax, mvaMin );
      cout << "Expression = " << expr << endl;
      tree->Project( "DataS", Form( "%s:%s", vary.Data(), varx.Data() ), 
                     Form( "%s*(type==1)", expr.Data() ) );
      tree->Project( "DataB", Form( "%s:%s", vary.Data(), varx.Data() ), 
                     Form( "%s*(type==0)", expr.Data() ) );
      tree->Project( "DataRS", Form( "%s:%s", vary.Data(), varx.Data() ), 
                     "type==1" );
      tree->Project( "DataRB", Form( "%s:%s", vary.Data(), varx.Data() ), 
                     "type==0" );
      tree->Project( "RefS", Form( "%s:%s", vary.Data(), varx.Data() ), 
                     "type==1", "", 500000  );
      tree->Project( "RefB", Form( "%s:%s", vary.Data(), varx.Data() ), 
                     "type==0", "", 500000, 10000 );

      Float_t zminS = frameS->GetMinimum();
      Float_t zmaxS = frameS->GetMaximum();
      Float_t zminB = frameB->GetMinimum();
      Float_t zmaxB = frameB->GetMaximum();
      // normalise      
      for (Int_t i=1; i<=nbin; i++) {
         for (Int_t j=1; j<=nbin; j++) {
            // signal
            Float_t z = frameS->GetBinContent( i, j );
            z = (z - zminS)/(zmaxS - zminS);
            Float_t zr = frameRS->GetBinContent( i, j );
            if (zr > 0) z /= zr;
            else z = 0.;
            frameS->SetBinContent( i, j, z );

            // background
            z = frameB->GetBinContent( i, j );
            z = (z - zminB)/(zmaxB - zminB);
            z = 1 - z;
            zr = frameRB->GetBinContent( i, j );
            if (zr > 0) z /= zr;
            else z = 0.;
            frameB->SetBinContent( i, j, z );
         }
      }
      zminS = frameS->GetMinimum();
      zmaxS = frameS->GetMaximum();
      zminB = frameB->GetMinimum();
      zmaxB = frameB->GetMaximum();
      // renormalise      
      for (Int_t i=1; i<=nbin; i++) {
         for (Int_t j=1; j<=nbin; j++) {
            // signal
            Float_t z = frameS->GetBinContent( i, j );
            z = 1*(z - zminS)/(zmaxS - zminS) - 0;
            frameS->SetBinContent( i, j, z );

            // background
            z = frameB->GetBinContent( i, j );
            z = 1*(z - zminB)/(zmaxB - zminB) - 0;
            frameB->SetBinContent( i, j, z );
         }
      }
      frame ->SetMinimum( -1.0 );
      frame ->SetMaximum( +1.0 );
      frameS->SetMinimum( -1.0 );
      frameS->SetMaximum( +1.0 );
      frameB->SetMinimum( -1.0 );
      frameB->SetMaximum( +1.0 );
      
      // axis labels
      frame->SetTitle( Form( "Signal and background distributions weighted by %s output", 
                             methodS.Data() ) );
      frame->SetTitleSize( 0.08 );
      frame->GetXaxis()->SetTitle( varx );
      frame->GetYaxis()->SetTitle( vary );

      // style
      frame->SetLabelSize( 0.04, "X" );
      frame->SetLabelSize( 0.04, "Y" );
      frame->SetTitleSize( 0.05, "X" );
      frame->SetTitleSize( 0.05, "Y" );
      frame->GetYaxis()->SetTitleOffset( 1.05);// label offset on x axis
      frame->GetYaxis()->SetTitleOffset( 1.30 );// label offset on x axis

      // now the weighted functions
      const Int_t nlevels = 3;
      Double_t levelsS[nlevels];
      Double_t levelsB[nlevels];
      levelsS[0] = 0.3;
      levelsS[1] = 0.5;
      levelsS[2] = 0.7;
      levelsB[0] = -0.3;
      levelsB[1] = 0.2;
      levelsB[2] = 0.5;
      frameS->SetContour( nlevels, levelsS );
      frameB->SetContour( nlevels, levelsB );

      frameS->SetLineColor( 104 );
      frameS->SetFillColor( 104 );
      frameS->SetLineWidth( 3 );
      frameB->SetLineColor( 102 );
      frameB->SetFillColor( 102 );
      frameB->SetLineWidth( 3 );

      // set style
      refS->SetMarkerSize( 0.2 );
      refS->SetMarkerColor( 104 );
      
      refB->SetMarkerSize( 0.2 );
      refB->SetMarkerColor( 102 );

      const Int_t nlevelsR = 1;
      Double_t levelsRS[nlevelsR];
      Double_t levelsRB[nlevelsR];
      levelsRS[0] = refS->GetMaximum()*0.3;
      //      levelsRS[1] = refS->GetMaximum()*0.3;
      levelsRB[0] = refB->GetMaximum()*0.3;
      //      levelsRB[1] = refB->GetMaximum()*0.3;
      refS->SetContour( nlevelsR, levelsRS );
      refB->SetContour( nlevelsR, levelsRB );

      refS->SetLineColor( 104 );
      refS->SetFillColor( 104 );
      refS->SetLineWidth( 3 );
      refB->SetLineColor( 102 );
      refB->SetFillColor( 102 );
      refB->SetLineWidth( 3 );

      // and plot
      c->cd(1);
      
      frame->Draw();
      frameS->Draw( "contsame" );
      refS->Draw( "cont3same" );
      refB->Draw( "cont3same" );  
      //      frameB->Draw( "colzsame" );

      // save canvas to file
      c->Update();
      if (Save_Images) {
         TMVAGlob::imgconv( c, Form("plots/mvaweights_%s",   methodS.Data()) );
      }
      countCanvas++;
   }
}
