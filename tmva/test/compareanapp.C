#include "tmvaglob.C"

void compareanapp( TString finAn = "TMVA.root", TString finApp = "TMVApp.root", bool useTMVAStyle=kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // switches
   const Bool_t Draw_CFANN_Logy = kFALSE;
   const Bool_t Save_Images     = kTRUE;

   TFile* file = TMVAGlob::OpenFile( finAn );  
   TFile* fileApp = TFile::Open( finApp, "READ" );  

   // define Canvas layout here!
   const Int_t width = 600;   // size of canvas

   // counter variables
   Int_t countCanvas = 0;
   char    fname[200];


   TIter next(file->GetListOfKeys());
   TKey *key, *hkey;
   char fname[200];
   while ((key = (TKey*)next())) {

      if (TString(key->GetClassName()) != "TDirectory" && TString(key->GetClassName()) != "TDirectoryFile") continue;
      if(! TString(key->GetName()).BeginsWith("Method_") ) continue;

      TDirectory * mDir = (TDirectory*)key->ReadObj();
      TIter nextInMDir(mDir->GetListOfKeys());
      while (hkey = (TKey*)nextInMDir()) {

         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(hkey->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *trainHist = (TH1*)hkey->ReadObj();
         TString hname= trainHist->GetName();
         
         if( !hname.BeginsWith("MVA_") ) continue;
         if( !hname.EndsWith("_S") ) continue;
         if( hname.Contains("muTransform") ) continue;

         TString hStem(hname);
         hStem.Remove(0,4); // removes "MVA_" in the beginning
         hStem.Remove(hStem.Length()-2); // removes the _S at the end



         // retrieve corresponding histogram from TMVApp.root   
         TH1* testHist = (TH1*)fileApp->Get( Form("MVA_%s", hStem.Data()) );
         if(testHist==0) {
            //            cout << "Error: Did not find histogram MVA_" << hStem << " in file " << fileApp->GetName() << endl;
            continue;
         } else {
            testHist->Scale( 1.0/testHist->Integral() );
         }

         //         cout << "--- Found training histogram: " << hname << endl;
         trainHist->Scale( 1.0/trainHist->Integral() );

         countCanvas++;
         c = new TCanvas( hStem, "MVA Output Variables", 
                          countCanvas*50, countCanvas*20, width, width*0.8 ); 

         // frame limits (choose judicuous x range)
         Float_t nrms = 4;
         // cout << "--- mean and RMS: " << trainHist->GetMean() << ", " << trainHist->GetRMS() << endl;
         Float_t xmin = TMath::Max( trainHist->GetMean() - nrms*trainHist->GetRMS(),
                                    trainHist->GetXaxis()->GetXmin() );
         Float_t xmax = TMath::Min( trainHist->GetMean() + nrms*trainHist->GetRMS(),
                                    trainHist->GetXaxis()->GetXmax() );
         Float_t ymin = 0;
         Float_t ymax = TMath::Max(trainHist->GetMaximum(),testHist->GetMaximum())*1.1 ;

         if (Draw_CFANN_Logy && hStem == "CFANN") {
            c->SetLogy();
            ymin = 0.01;
         }

         // build a frame
         Int_t nb = 500;
         TH2F* frame = new TH2F( TString("frame") + trainHist->GetName(), trainHist->GetTitle(), 
                                 nb, xmin, xmax, nb, ymin, ymax );
         frame->SetTitle( TString("MVA output for method: ") + hStem );
         frame->GetYaxis()->SetTitle("Normalized");
         TMVAGlob::SetFrameStyle( frame );

         // eventually: draw the frame
         frame->Draw();  

         testHist->SetLineWidth( 2 );
         //         trainHist->Draw("same hist");
         testHist->Draw("same hist");

         // save canvas to file
         TMVAGlob::plot_logo();
         c->Update();
         if (Save_Images) TMVAGlob::imgconv( c, Form("plots/compareanapp_c%i", countCanvas) );
      }
   }
}
