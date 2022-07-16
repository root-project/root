#include "TMVA/rulevisHists.h"
#include "TH2.h"


// This macro plots the distributions of the different input variables overlaid on
// the sum of importance per bin.
// The scale goes from violett (no importance) to red (high importance).
// Areas where many important rules are active, will thus be very red.
//
// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void TMVA::rulevisHists(TString fin, TMVAGlob::TypeOfPlot type, bool useTMVAStyle)
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
   
   // checks if file with name "fin" is already open, and if not opens one
   //TFile *file =
   TMVAGlob::OpenFile( fin );

   // get all titles of the method rulefit
   TList titles;
   TString dirname = "Method_RuleFit";
   UInt_t ninst = TMVAGlob::GetListOfTitles(dirname,titles);
   if (ninst==0) return;

   // get top dir containing all hists of the variables
   TDirectory* vardir = TMVAGlob::GetInputVariablesDir( type );
   if (vardir==0) return;

   TDirectory* corrdir = TMVAGlob::GetCorrelationPlotsDir( type, vardir );
   if (corrdir==0) return;

   // loop over all titles
   TIter keyIter(&titles);
   TDirectory *rfdir;
   TKey *rfkey;
   while ((rfkey = TMVAGlob::NextKey(keyIter,"TDirectory"))) {
      rfdir = (TDirectory *)rfkey->ReadObj();
      rulevisHists( rfdir, vardir, corrdir, type );
   }
}

void TMVA::rulevisHists( TDirectory *rfdir, TDirectory *vardir, TDirectory *corrdir, TMVAGlob::TypeOfPlot type) {
   //
   if (rfdir==0)   return;
   if (vardir==0)  return;
   if (corrdir==0) return;
   //
   const TString rfName    = rfdir->GetName();
   const TString maintitle = rfName + " : Rule Importance";
   const TString rfNameOpt = "_RF2D_";
   const TString outfname[TMVAGlob::kNumOfMethods] = { "rulevisHists",
                                                       "rulevisHists_decorr",
                                                       "rulevisCorr_pca",
                                                       "rulevisCorr_gaussdecorr" };

   const TString outputName = outfname[type]+"_"+rfdir->GetName();
   //
   TIter rfnext(rfdir->GetListOfKeys());
   TKey *rfkey;
   Double_t rfmax=0;//keep compiler quiet .. it IS in any case
   Double_t rfmin=0;// later initialized
   //   Bool_t allEmpty=kTRUE;
   Bool_t first=kTRUE;
   while ((rfkey = (TKey*)rfnext())) {
      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(rfkey->GetClassName());
      if (!cl->InheritsFrom("TH2F")) continue;
      TH2F *hrf = (TH2F*)rfkey->ReadObj();
      TString hname= hrf->GetName();
      if (hname.Contains("__RF_")){ // found a new RF plot
         Double_t valmin = hrf->GetMinimum();
         Double_t valmax = hrf->GetMaximum();
         if (first) {
            rfmin=valmin;
            rfmax=valmax;
            first = kFALSE;
         } else {
            if (valmax>rfmax) rfmax=valmax;
            if (valmin<rfmin) rfmin=valmin;
         }
         //         if (hrf->GetEntries()>0) allEmpty=kFALSE;
      }
   }
   if (first) {
      cout << "ERROR: no RF plots found..." << endl;
      return;
   }

   const Int_t nContours = 100;
   Double_t contourLevels[nContours];
   Double_t dcl = (rfmax-rfmin)/Double_t(nContours-1);
   //
   for (Int_t i=0; i<nContours; i++) {
      contourLevels[i] = rfmin+dcl*Double_t(i);
   }

   ///////////////////////////
   vardir->cd();
 
   // how many plots are in the directory?
   Int_t noPlots = ((vardir->GetListOfKeys())->GetEntries()) / 2;
 
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
   TIter next(vardir->GetListOfKeys());
   TKey *key;
   //
   first = kTRUE;

   while ((key = (TKey*)next())) {

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1")) continue;
      TH1F* sig = (TH1F*)key->ReadObj();
      TString hname= sig->GetName();

      // check for all signal histograms
      if (hname.Contains("__S")){ // found a new signal plot
         // create new canvas
         if ((c[countCanvas]==NULL) || (countPad>noPad)) {
            char cn[20];
            snprintf( cn, 20, "rulehist%d_", countCanvas+1 );
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
         bgname.ReplaceAll("__S","__B");
         TKey *hkey = vardir->GetKey(bgname);
         TH1F* bgd = (TH1F*)hkey->ReadObj();
         if (bgd == NULL) {
            cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
            //exit(1);
            delete [] c;
            return;
         }

         TString rfname = hname;
         rfname.ReplaceAll("__S","__RF");
         TKey *hrfkey = rfdir->GetKey(rfname);
         TH2F *hrf = (TH2F*)hrfkey->ReadObj();
         //         Double_t wv = hrf->GetMaximum();
         //         if (rfmax>0.0)
         //            hrf->Scale(1.0/rfmax);
         hrf->SetMinimum(rfmin); // make sure it's zero  -> for palette axis
         hrf->SetMaximum(rfmax); // make sure max is 1.0 -> idem
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

         if (first) {
            hrf->SetTitle( maintitle );
            first = kFALSE;
         } else {
            hrf->SetTitle( "" );
         }
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
            TString fname = Form( "plots/%s_c%i", outputName.Data(), countCanvas+1 );
            TMVAGlob::imgconv( c[countCanvas], fname );
            //        TMVAGlob::plot_logo(); // don't understand why this doesn't work ... :-(
            countCanvas++;
         }
      }
   }

   if (countPad <= noPad) {
      c[countCanvas]->Update();
      TString fname = Form( "plots/%s_c%i", outputName.Data(), countCanvas+1 );
      TMVAGlob::imgconv( c[countCanvas], fname );
   }
   delete[] c;
}
