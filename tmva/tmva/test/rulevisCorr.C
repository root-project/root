#include "tmvaglob.C"

// This macro plots the distributions of the different input variables overlaid on
// the sum of importance per bin.
// The scale goes from violett (no importance) to red (high importance).
// Areas where many important rules are active, will thus be very red.
//
// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void rulevisCorr( TString fin = "TMVA.root", TMVAGlob::TypeOfPlot type = TMVAGlob::kNorm, bool useTMVAStyle=kTRUE )
{

   // set style and remove existing canvas'
   //   TMVAGlob::Initialize( useTMVAStyle );
   
   // checks if file with name "fin" is already open, and if not opens one
   TFile *file = TMVAGlob::OpenFile( fin );

   // get top dir containing all hists of the variables
//    TDirectory* vardir = (TDirectory*)file->Get( "InputVariables_Id" );
//    TDirectory* vardir = TMVAGlob::GetInputVariablesDir( type );
//    if (vardir==0) return;
   
//    TDirectory* corrdir = TMVAGlob::GetCorrelationPlotsDir( type, vardir );
//    if (corrdir==0) return;

   // get all titles of the method rulefit
   TList titles;
   UInt_t ninst = TMVAGlob::GetListOfTitles("Method_RuleFit",titles);
   if (ninst==0) return;

   TDirectory* dir = (TDirectory*)file->Get( "Method_RuleFit" );

   // loop over rulefit methods 
   TIter next(dir->GetListOfKeys());
   TKey *key(0);   
   while ((key = (TKey*)next())) {

      if (!gROOT->GetClass(key->GetClassName())->InheritsFrom("TDirectory")) continue;

      TDirectory* rfdir   = (TDirectory*)key->ReadObj();      
      TDirectory* vardir  = rfdir;
      TDirectory* corrdir = rfdir;
      
      // loop over all titles
      TIter keyIter(&titles);
      TDirectory *rfdir;
      TKey *rfkey;
      //      while ((rfkey = TMVAGlob::NextKey(keyIter,"TDirectory"))) {
      //         rfdir = (TDirectory *)rfkey->ReadObj();
         rulevisCorr( rfdir, vardir, corrdir, type );
         //      }
   }
}

void rulevisCorr( TDirectory *rfdir, TDirectory *vardir, TDirectory *corrdir, TMVAGlob::TypeOfPlot type) {
   //
   if (rfdir==0)   return;
   if (vardir==0)  return;
   if (corrdir==0) return;
   //
   const TString rfName    = rfdir->GetName();
   const TString maintitle = rfName + " : Rule Importance, 2D";
   const TString rfNameOpt = "_RF2D_";
   const TString outfname[TMVAGlob::kNumOfMethods] = { "rulevisCorr",
                                                       "rulevisCorr_decorr",
                                                       "rulevisCorr_pca",
                                                       "rulevisCorr_gaussdecorr" };
   const TString outputName = outfname[type]+"_"+rfdir->GetName();
   //
   TIter rfnext(rfdir->GetListOfKeys());
   TKey *rfkey;
   Double_t rfmax = -1;
   Double_t rfmin = -1;
   Bool_t allEmpty=kTRUE;
   Bool_t first=kTRUE;
   TH2F *hrf;
   while ((rfkey = TMVAGlob::NextKey(rfnext,"TH2F"))) {
      hrf = (TH2F *)rfkey->ReadObj();
      TString hname= hrf->GetName();
      if (hname.Contains(rfNameOpt)){ // found a new RF2D plot
         Double_t valmin = hrf->GetMinimum();
         Double_t valmax = hrf->GetMaximum();
         if (first) {
            rfmax = valmax;
            rfmin = valmin;
         } else {
            if (valmax>rfmax) rfmax = valmax;
            if (valmin<rfmin) rfmin = valmin;
         }
         if (hrf->GetEntries()>0) allEmpty=kFALSE;
         first=kFALSE;
      }
   }
   if (first) {
      cout << "ERROR: no RF2D plots found..." << endl;
      return;
   }
   Double_t minrange   = rfmin;
   Double_t maxrange   = rfmax;
   Double_t targetrange = maxrange - minrange;

   const Int_t nContours = 100;
   Double_t contourLevels[nContours];
   Double_t dcl = targetrange/Double_t(nContours-1);
   //
   for (Int_t i=0; i<nContours; i++) {
      contourLevels[i] = rfmin+dcl*Double_t(i);
   }

   ///////////////////////////
   vardir->cd();
 
   // how many plots are in the directory?
   Int_t noVars = ((vardir->GetListOfKeys())->GetEntries()) / 2;
   Int_t noPlots = (noVars*(noVars+1)/2) - noVars;

   // *** CONTINUE HERE *** 
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

   // counter variables
   Int_t countCanvas = 0;
   Int_t countPad    = 1;

   // loop over all objects in directory
   TIter next(corrdir->GetListOfKeys());
   TKey *key;
   TH2F *sigCpy=0;
   TH2F *bgdCpy=0;
   Bool_t first = kTRUE;
   //
   while ((key = (TKey*)next())) {

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH2")) continue;
      sig = (TH2F*)key->ReadObj();
      TString hname= sig->GetName();
      // check for all signal histograms
      if (hname.Contains("_sig_")){ // found a new signal plot

         // create new canvas
         if ((c[countCanvas]==NULL) || (countPad>noPad)) {
            char cn[20];
            sprintf( cn, "rulecorr%d_", countCanvas+1 );
            TString cname(cn);
            cname += rfdir->GetName();
            c[countCanvas] = new TCanvas( cname, maintitle,
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
         bgname.ReplaceAll("_sig_","_bgd_");
         hkey = corrdir->GetKey(bgname);
         bgd = (TH2F*)hkey->ReadObj();
         if (bgd == NULL) {
            cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
            exit;
         }
         const Int_t rebin=6;
         sig->Rebin2D(rebin,rebin);
         bgd->Rebin2D(rebin,rebin);
         //
         TString rfname = hname;
         rfname.ReplaceAll("_sig_",rfNameOpt);
         TKey *hrfkey = rfdir->GetKey(rfname);
         hrf2 = (TH2F*)hrfkey->ReadObj();
         Double_t wmin = hrf2->GetMinimum();
         Double_t wmax = hrf2->GetMaximum();
         Double_t wmean  = (wmax+wmin)/2.0;
         Double_t wrange = wmax-wmin;
         Double_t wscale = (wrange>0.0 ? targetrange/wrange : 1.0);
         //         if (rfmax>0.0)
         //            hrf2->Scale(1.0/rfmax);
         hrf2->SetMinimum(minrange); // make sure it's zero  -> for palette axis
         hrf2->SetMaximum(maxrange); // make sure max is 1.0 -> idem
         hrf2->SetContour(nContours,&contourLevels[0]);

         // this is set but not stored during plot creation in MVA_Factory
         //         TMVAGlob::SetSignalAndBackgroundStyle( sigK, bgd );
         sig->SetFillColor(1);
         sig->SetLineColor(1);

         bgd->SetFillColor(15);
         bgd->SetLineColor(15);

         // chop off "signal" 
         TString title(hrf2->GetTitle());
         title.ReplaceAll("signal","");
         if (first) {
            hrf2->SetTitle( maintitle );
            first=kFALSE;
         } else {
            hrf2->SetTitle( "" );
         }
         TMVAGlob::SetFrameStyle( hrf2, 1.2 );

         // finally plot and overlay       
         hrf2->Draw("colz ah");
         Float_t sc = 1.1;
         if (countPad==2) sc = 1.3;
         sig->SetMaximum( TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*sc );
         Double_t smax = sig->GetMaximum();

         sig->Scale(1.0/smax);
         sig->SetContour(5);
         sig->Draw("same cont3");
         TMVAGlob::SetFrameStyle( sig, 1.2 );

         bgd->Scale(1.0/smax);
         bgd->SetContour(5);
         bgd->Draw("same cont3");
         TMVAGlob::SetFrameStyle( bgd, 1.2 );
         //         sig->GetXaxis()->SetTitle( title );
         sig->GetYaxis()->SetTitleOffset( 1.30 );
         //         sig->GetYaxis()->SetTitle("Events");

         // redraw axes
         sig->Draw("sameaxis");

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
            legend->SetFillStyle(3001);
         } 

         // save canvas to file
         if (countPad > noPad) {
            c[countCanvas]->Update();
            TString fname = Form( "plots/%s_c%i", outputName.Data(), countCanvas+1 );
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
