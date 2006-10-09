#include "tmvaglob.C"

void annconvergencetest( TString fin = "TMVA.root", bool useTMVAStyle=kTRUE )
{
  if (!useTMVAStyle) {
    gROOT->Reset();
    gROOT->SetStyle("Plain");
    gStyle->SetOptStat(0);
  }

  TList* loc = gROOT->GetListOfCanvases();
  TListIter itc(loc);
  TObject *o(0);
  while( (o = itc()) ) delete o;

  TFile *file = new TFile( fin );

  TDirectory * dir = (TDirectory*)gDirectory->Get("MLP");
  if (dir==0) {
    cout << "Could not locate directory MLP in file " << fin << endl;
    return;
  }
  dir->cd();  

  TCanvas* c = new TCanvas( "MLPConvergenceTest", "MLP Convergence Test", 100, 0, 600, 580*0.8 ); 

//   Float_t newMargin1 = 0.13;
//   Float_t newMargin2 = 0.12;

//   gPad->SetTicks();
//   gPad->SetLeftMargin  ( newMargin2 );
//   gPad->SetBottomMargin( newMargin2 );
//   gPad->SetRightMargin ( 0.04 );
//   gPad->SetTopMargin   ( newMargin1 );
  
  Double_t m1  = estimatorHistTrain->GetMaximum();
  Double_t m2  = estimatorHistTest ->GetMaximum();
  Double_t max = TMath::Max( m1, m2 );
  m1  = estimatorHistTrain->GetMinimum();
  m2  = estimatorHistTest ->GetMinimum();
  Double_t min = TMath::Min( m1, m2 );
  estimatorHistTrain->SetMaximum( max + 0.1*(max - min) );
  estimatorHistTrain->SetMinimum( min - 0.1*(max - min) );
  estimatorHistTrain->SetLineColor( 2 );
  estimatorHistTrain->SetLineWidth( 2 );
  estimatorHistTrain->SetTitle( TString("MLP Convergence test") );
  
  estimatorHistTest->SetLineColor( 4 );
  estimatorHistTest->SetLineWidth( 2 );

  estimatorHistTrain->GetXaxis()->SetTitle( "Epochs" );
  estimatorHistTrain->GetYaxis()->SetTitle( "Estimator" );
  estimatorHistTrain->GetXaxis()->SetTitleOffset( 1.20 );
  estimatorHistTrain->GetYaxis()->SetTitleOffset( 1.65 );

  estimatorHistTrain->Draw();
  estimatorHistTest ->Draw("same");

  // need a legend
  TLegend *legend= new TLegend( 1 - gPad->GetRightMargin() - 0.45, 1-gPad->GetTopMargin() - 0.20, 
				1 - gPad->GetRightMargin() - 0.05, 1-gPad->GetTopMargin() - 0.05 );

  legend->AddEntry(estimatorHistTrain,"Training Sample","l");
  legend->AddEntry(estimatorHistTest,"Test sample","l");
  legend->Draw("same");
  legend->SetBorderSize(1);
  legend->SetMargin( 0.3 );

  c->cd();
  TMVAGlob::plot_logo(); // don't understand why this doesn't work ... :-(
  c->Update();

  TString fname = "plots/annconvergencetest";
  TMVAGlob::imgconv( c, fname );
}
