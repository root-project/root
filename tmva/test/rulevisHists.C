#include "tmvaglob.C"

// This macro plots the distributions of the different input variables overlaid on
// the sum of importance per bin.
// The scale goes from violett (no importance) to red (high importance).
// Areas where many important rules are active, will thus be very red.
//
// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void rulevisHists( TString fin = "TMVA.root", TMVAGlob::TypeOfPlot type = TMVAGlob::kNormal, bool useTMVAStyle=kTRUE )
{
   const TString rulefitdir = "Method_RuleFit";

   const TString directories[TMVAGlob::kNumOfMethods] = { "InputVariables_NoTransform",
                                                          "InputVariables_DecorrTransform",
                                                          "InputVariables_PCATransform" };
   
   const TString outfname[TMVAGlob::kNumOfMethods] = { "rulevisHists",
                                                       "rulevisHists_decorr",
                                                       "rulevisHists_pca" };

   const TString maintitle = "Rule importance";

   const Int_t nContours = 100;
   Double_t contourLevels[nContours];
   Double_t dcl = 1.0/Double_t(nContours-1);
   //
   for (Int_t i=0; i<nContours; i++) {
      contourLevels[i] = dcl*Double_t(i);
   }

   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
   
   // checks if file with name "fin" is already open, and if not opens one
   TFile *file = TMVAGlob::OpenFile( fin );

   TDirectory* rfdir = (TDirectory*)gDirectory->Get( rulefitdir );
   if (rfdir==0) {
      cout << "Could not locate directory '" << rulefitdir << "' in file: " << fin << endl;
      return;
   }
   
   TDirectory* dir = (TDirectory*)gDirectory->Get( directories[type] );
   if (dir==0) {
      cout << "Could not locate directory '" << directories[type] << "' in file: " << fin << endl;
      return;
   }

   TIter rfnext(rfdir->GetListOfKeys());
   TKey *rfkey;
   Double_t rfmax=-1.0;
   Bool_t allEmpty=kTRUE;
   while ((rfkey = (TKey*)rfnext())) {
      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(rfkey->GetClassName());
      if (!cl->InheritsFrom("TH2F")) continue;
      TH2F *hrf = (TH2F*)rfkey->ReadObj();
      TString hname= hrf->GetName();
      if (hname.Contains("__RF_")){ // found a new RF plot
         Double_t val = hrf->GetMaximum();
         if (val>rfmax) rfmax=val;
         if (hrf->GetEntries()>0) allEmpty=kFALSE;
      }
   }
   if (rfmax<0) {
      cout << "ERROR: no RF plots found..." << endl;
      return;
   }
   ///////////////////////////
   dir->cd();
 
   // how many plots are in the directory?
   Int_t noPlots = ((dir->GetListOfKeys())->GetEntries()) / 2;
 
   // define Canvas layout here!
   // default setting
   Int_t xPad;  // no of plots in x
   Int_t yPad;  // no of plots in y
   Int_t width; // size of canvas
   Int_t height;
   switch (noPlots) {
   case 1:
      xPad = 1; yPad = 1; width = 500; height = 0.7*width; break;
   case 2:
      xPad = 2; yPad = 1; width = 600; height = 0.7*width; break;
   case 3:
      xPad = 3; yPad = 1; width = 900; height = 0.4*width; break;
   case 4:
      xPad = 2; yPad = 2; width = 600; height = width; break;
   default:
      xPad = 3; yPad = 2; width = 800; height = 0.7*width; break;
   }
   Int_t noPad = xPad * yPad ;   

   // this defines how many canvases we need
   const Int_t noCanvas = 1 + (Int_t)((noPlots - 0.001)/noPad);
   TCanvas **c = new TCanvas*[noCanvas];
   for (Int_t ic=0; ic<noCanvas; ic++) c[ic] = 0;

   cout << "--- Found: " << noPlots << " plots; will produce: " << noCanvas << " canva(s)" << endl;

   // counter variables
   Int_t countCanvas = 0;
   Int_t countPad    = 1;

   // loop over all objects in directory
   TIter next(dir->GetListOfKeys());
   TKey *key;
   TH1F *sigCpy=0;
   TH1F *bgdCpy=0;
   //
   while ((key = (TKey*)next())) {

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1")) continue;
      sig = (TH1F*)key->ReadObj();
      TString hname= sig->GetName();

      // check for all signal histograms
      if (hname.Contains("__S")){ // found a new signal plot
         //         sigCpy = new TH1F(*sig);
         // create new canvas
         if ((c[countCanvas]==NULL) || (countPad>noPad)) {
            cout << "--- Book canvas no: " << countCanvas << endl;
            char cn[20];
            sprintf( cn, "canvas%d", countCanvas+1 );
            c[countCanvas] = new TCanvas( cn, maintitle,
                                          countCanvas*50+200, countCanvas*20, width, height ); 
            // style
            c[countCanvas]->Divide(xPad,yPad);
            countPad = 1;
         }       

         // save canvas to file
         TPad *cPad = (TPad *)(c[countCanvas]->GetPad(countPad));
         c[countCanvas]->cd(countPad);
         countPad++;

         // find the corredponding background histo
         TString bgname = hname;
         bgname.ReplaceAll("__S","__B");
         hkey = dir->GetKey(bgname);
         bgd = (TH1F*)hkey->ReadObj();
         if (bgd == NULL) {
            cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
            exit;
         }

         TString rfname = hname;
         rfname.ReplaceAll("__S","__RF");
         TKey *hrfkey = rfdir->GetKey(rfname);
         TH2F *hrf = (TH2F*)hrfkey->ReadObj();
         Double_t wv = hrf->GetMaximum();
         if (rfmax>0.0)
            hrf->Scale(1.0/rfmax);
         hrf->SetMinimum(0.0); // make sure it's zero  -> for palette axis
         hrf->SetMaximum(1.0); // make sure max is 1.0 -> idem
         hrf->SetContour(nContours,&contourLevels[0]);

         // this is set but not stored during plot creation in MVA_Factory
         //         TMVAGlob::SetSignalAndBackgroundStyle( sigK, bgd );
         sig->SetFillStyle(3002);
         sig->SetFillColor(1);
         sig->SetLineColor(1);
         sig->SetLineWidth(2);

         bgd->SetFillStyle(3554);
         bgd->SetFillColor(1);
         bgd->SetLineColor(1);
         bgd->SetLineWidth(2);

         // chop off "signal" 
         TString title(hrf->GetTitle());
         title.ReplaceAll("signal","");

         // finally plot and overlay       
         Float_t sc = 1.1;
         if (countPad==2) sc = 1.3;
         sig->SetMaximum( TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*sc );
         Double_t smax = sig->GetMaximum();

         hrf->SetTitle( TString( maintitle ) + ": " + title );
         hrf->Draw("colz ah");
         TMVAGlob::SetFrameStyle( hrf, 1.2 );

         sig->Draw("same ah");
         bgd->Draw("same ah");
         // draw axis using range [0,smax]
         hrf->GetXaxis()->SetTitle( title );
         hrf->GetYaxis()->SetTitleOffset( 1.30 );
         hrf->GetYaxis()->SetTitle("Events");
         hrf->GetYaxis()->SetLimits(0,smax);
         hrf->Draw("same axis");

         cPad->SetRightMargin(0.13);
         cPad->Update();

         // Draw legend
         if (countPad==2){
            TLegend *legend= new TLegend( cPad->GetLeftMargin(), 
                                          1-cPad->GetTopMargin()-.18, 
                                          cPad->GetLeftMargin()+.4, 
                                          1-cPad->GetTopMargin() );
            legend->AddEntry(sig,"Signal","F");
            legend->AddEntry(bgd,"Background","F");
            legend->Draw("same");
            legend->SetBorderSize(1);
            legend->SetMargin( 0.3 );
            legend->SetFillColor(19);
            legend->SetFillStyle(1);
         } 

         // save canvas to file
         if (countPad > noPad) {
            c[countCanvas]->Update();
            TString fname = Form( "plots/%s_c%i", outfname[type].Data(), countCanvas+1 );
            TMVAGlob::imgconv( c[countCanvas], fname );
            //        TMVAGlob::plot_logo(); // don't understand why this doesn't work ... :-(
            countCanvas++;
         }
      }
   }

   if (countPad <= noPad) {
      c[countCanvas]->Update();
      TString fname = Form( "plots/%s_c%i", outfname[type].Data(), countCanvas+1 );
      TMVAGlob::imgconv( c[countCanvas], fname );
   }
}
