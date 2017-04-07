#include "TMVA/annconvergencetest.h"


// this macro serves to assess the convergence of the MLP ANN. 
// It compares the error estimator for the training and testing samples.
// If overtraining occurred, the estimator for the training sample should 
// monotoneously decrease, while the estimator of the testing sample should 
// show a minimum after which it increases.

// input: - Input file (result from TMVA),
//        - use of TMVA plotting TStyle

void TMVA::annconvergencetest(TString dataset, TDirectory *lhdir )
{
   TString jobName = lhdir->GetName();
   static int icanvas = -1;
   icanvas++;
   TCanvas* c = new TCanvas( Form("MLPConvergenceTest_%s",jobName.Data()), Form("MLP Convergence Test, %s",jobName.Data()), 
                             100 + (icanvas)*40, 0 + (icanvas+1)*20, 600, 580*0.8  );
  
   TH1* estimatorHistTrain = (TH1*)lhdir->Get( "estimatorHistTrain" );
   TH1* estimatorHistTest  = (TH1*)lhdir->Get( "estimatorHistTest"  );

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
   estimatorHistTrain->SetTitle( TString("MLP Convergence Test") );
  
   estimatorHistTest->SetLineColor( 4 );
   estimatorHistTest->SetLineWidth( 2 );

   estimatorHistTrain->GetXaxis()->SetTitle( "Epochs" );
   estimatorHistTrain->GetYaxis()->SetTitle( "Estimator" );
   estimatorHistTrain->GetXaxis()->SetTitleOffset( 1.20 );
   estimatorHistTrain->GetYaxis()->SetTitleOffset( 1.65 );

   estimatorHistTrain->Draw("hist");
   estimatorHistTest ->Draw("samehist");

   // need a legend
   TLegend *legend= new TLegend( 1 - c->GetRightMargin() - 0.45, 1-c->GetTopMargin() - 0.20, 
                                 1 - c->GetRightMargin() - 0.05, 1-c->GetTopMargin() - 0.05 );

   legend->AddEntry(estimatorHistTrain,"Training Sample","l");
   legend->AddEntry(estimatorHistTest,"Test sample","l");
   legend->Draw("same");
   legend->SetMargin( 0.3 );

   c->cd();
   TMVAGlob::plot_logo(); // don't understand why this doesn't work ... :-(
   c->Update();

   TString fname = dataset+"/plots/annconvergencetest";
   TMVAGlob::imgconv( c, fname );
}

void TMVA::annconvergencetest(TString dataset, TString fin , Bool_t useTMVAStyle  )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
  
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // get all titles of the method likelihood
   TList titles;
   TString metmlp="Method_MLP"; 
   UInt_t ninst = TMVAGlob::GetListOfTitles(metmlp,titles,file->GetDirectory(dataset.Data()));
   if (ninst==0) {
      cout << "Could not locate directory 'Method_MLP' in file " << fin << endl;
      return;
   }
   // loop over all titles
   TIter keyIter(&titles);
   TDirectory *lhdir;
   TKey *key;
   while ((key = TMVAGlob::NextKey(keyIter,"TDirectory"))) {
      lhdir = (TDirectory *)key->ReadObj();
      annconvergencetest(dataset, lhdir );
   }
}
