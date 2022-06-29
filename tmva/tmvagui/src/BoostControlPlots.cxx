#include "TMVA/BoostControlPlots.h"
#include <vector>
#include <string>
#include "TLegend.h"
#include "TText.h"



// input: - Input file (result from TMVA),
//        - use of TMVA plotting TStyle
// this macro is based on BDTControlPlots.C
void TMVA::BoostControlPlots(TString dataset, TString fin , Bool_t useTMVAStyle  )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
  
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   // get all titles of the method Boost
   TList titles;
   TString dirname="Method_Boost";
   UInt_t ninst = TMVA::TMVAGlob::GetListOfTitles(dirname,titles,file->GetDirectory(dataset.Data()));
   if (ninst==0) {
      cout << "Could not locate directory 'Method_Boost' in file " << fin << endl;
      return;
   }
   // loop over all titles
   TIter keyIter(&titles);
   TDirectory *boostdir;
   TKey *key;
   while ((key = TMVAGlob::NextKey(keyIter,"TDirectory"))) {
      boostdir = (TDirectory *)key->ReadObj();
      boostcontrolplots(dataset, boostdir );
   }
}

void TMVA::boostcontrolplots(TString dataset, TDirectory *boostdir ) {

   const Int_t nPlots = 6;

   Int_t width  = 900;
   Int_t height = 900;
   char cn[100];
   const TString titName = boostdir->GetName();
   sprintf( cn, "cv_%s", titName.Data() );
   TCanvas *c = new TCanvas( cn,  Form( "%s Control Plots", titName.Data() ),
                             width, height ); 
   c->Divide(2,4);


   //TString hname[nPlots]={"Booster_BoostWeight","Booster_MethodWeight","Booster_ErrFraction","Booster_OrigErrFraction"};

   TString hname[nPlots]={"BoostWeight","MethodWeight","ErrFraction","SoverBtotal","SeparationGain", "SeparationGain"};

   //   Note: the ROCIntegral plots are only filled for option "Boost_DetailedMonitoring=ture" currently not filled...
   //   TString hname[nPlots]={"BoostWeight","MethodWeight","ErrFraction","ROCIntegral_test"};

   for (Int_t i=0; i<nPlots; i++){
      Int_t color = 4; 
      TH1 *h = (TH1*) boostdir->Get(hname[i]);
      TString plotname = h->GetName();
      h->SetMaximum(h->GetMaximum()*1.3);
      h->SetMinimum( 0 );
      h->SetMarkerColor(color);
      h->SetMarkerSize( 0.7 );
      h->SetMarkerStyle( 24 );
      h->SetLineWidth(2);
      h->SetLineColor(color);
      h->Draw();
      c->Update();
   }

   // draw combined ROC plots

   TString hname_roctest[2] ={"ROCIntegral_test",  "ROCIntegralBoosted_test"};
   TString hname_roctrain[2]={"ROCIntegral_train", "ROCIntegralBoosted_train"};
   TString htitle[2] = {"ROC integral of single classifier", "ROC integral of boosted method"};

   for (Int_t i=0; i<2; i++){
      Int_t color = 4; 
      TPad * cPad = (TPad*)c->cd(nPlots+i+1);
      TH1 *htest  = (TH1*) boostdir->Get(hname_roctest[i]);
      TH1 *htrain = (TH1*) boostdir->Get(hname_roctrain[i]);

      // check if filled 
      //      Bool_t histFilled = (htest->GetMaximum() > 0 || htrain->GetMaximum() > 0);
      Bool_t histFilled = (htest && htrain);

      if (!htest)  htest  = new TH1F("htest","",2,0,1);
      if (!htrain) htrain = new TH1F("htrain","",2,0,1);

      htest->SetTitle(htitle[i]);
      htest->SetMaximum(1.0);
      htest->SetMinimum(0.0);
      htest->SetMarkerColor(color);
      htest->SetMarkerSize( 0.7 );
      htest->SetMarkerStyle( 24 );
      htest->SetLineWidth(2);
      htest->SetLineColor(color);
      htest->Draw();
      htrain->SetMaximum(1.0);
      htrain->SetMinimum(0.0);
      htrain->SetMarkerColor(color-2);
      htrain->SetMarkerSize( 0.7 );
      htrain->SetMarkerStyle( 24 );
      htrain->SetLineWidth(2);
      htrain->SetLineColor(color-2);
      htrain->Draw("same");

      if (histFilled) {
         TLegend *legend= new TLegend( cPad->GetLeftMargin(), 
                                       0.2 + cPad->GetBottomMargin(),
                                       cPad->GetLeftMargin() + 0.6, 
                                       cPad->GetBottomMargin() );
         legend->AddEntry(htest,  TString("testing sample"),  "L");
         legend->AddEntry(htrain, TString("training sample (orig. weights)"), "L");
         legend->SetFillStyle( 1 );
         legend->SetBorderSize(1);
         legend->SetMargin( 0.3 );
         legend->Draw("same");
      }
      else {
         TText t;
         t.SetTextSize( 0.056 );
         t.SetTextColor( 2 );
         t.DrawTextNDC( .2, 0.6, "Use MethodBoost option: \"Boost_DetailedMonitoring\" " );        
         t.DrawTextNDC( .2, 0.51, "to fill this histograms" );        
      }

      c->Update();
   }

   // write to file
   TString fname = dataset+Form( "/plots/%s_ControlPlots", titName.Data() );
   TMVAGlob::imgconv( c, fname );

}


