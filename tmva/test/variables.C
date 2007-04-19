#include "tmvaglob.C"

// this macro plots the distributions of the different input variables
// used in TMVA (e.g. running TMVAnalysis.C).  Signal and Background are overlayed.

// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void variables( TString fin = "TMVA.root", TMVAGlob::TypeOfPlot type = TMVAGlob::kNormal, bool useTMVAStyle=kTRUE )
{

   const TString directories[3] = { "InputVariables_NoTransform",
                                    "InputVariables_DecorrTransform",
                                    "InputVariables_PCATransform" };

   const TString titles[3] = { "TMVA Input Variable",
                               "Decorrelated TMVA Input Variables",
                               "Principal Component Transformed TMVA Input Variables" };

   const TString outfname[3] = { "variables",
                                 "variables_decorr",
                                 "variables_pca" };


   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );

   TDirectory* dir = (TDirectory*)file->Get( directories[type] );
   if (dir==0) {
      cout << "No information about " << titles[type] << " available in " << fin << endl;
      return;
   }
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
      xPad = 1; yPad = 1; width = 500; height = 0.55*width; break;
   case 2:
      xPad = 2; yPad = 1; width = 600; height = 0.55*width; break;
   case 3:
      xPad = 3; yPad = 1; width = 900; height = 0.4*width; break;
   case 4:
      xPad = 2; yPad = 2; width = 600; height = width; break;
   default:
      xPad = 3; yPad = 2; width = 800; height = 0.55*width; break;
   }
   Int_t noPadPerCanv = xPad * yPad ;

   // counter variables
   Int_t countCanvas = 0;
   Int_t countPad    = 0;

   // loop over all objects in directory
   TIter next(dir->GetListOfKeys());
   TKey    * key  = 0;
   TCanvas * canv = 0;
   while ((key = (TKey*)next())) {
      if (key->GetCycle() != 1) continue;

      if(! TString(key->GetName()).Contains("__S")) continue;

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1")) continue;
      TH1 *sig = (TH1*)key->ReadObj();
      TString hname(sig->GetName());

      // create new canvas
      if (countPad%noPadPerCanv==0) {
         ++countCanvas;
         canv = new TCanvas( Form("canvas%d", countCanvas), titles[type],
                             countCanvas*50+200, countCanvas*20, width, height );
         canv->Divide(xPad,yPad);
         canv->Draw();
      }

      TPad * cPad = (TPad*)canv->cd(countPad++%noPadPerCanv+1);
      
      // find the corredponding backgrouns histo
      TString bgname = hname;
      bgname.ReplaceAll("__S","__B");
      TH1 *bgd = (TH1*)dir->Get(bgname);
      if (bgd == NULL) {
         cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
         exit;
      }

      // this is set but not stored during plot creation in MVA_Factory
      TMVAGlob::SetSignalAndBackgroundStyle( sig, bgd );

      // chop off "signal"
      TString title(sig->GetTitle());
      title.ReplaceAll("signal","");
      sig->SetTitle( TString( titles[type] ) + ": " + title );
      TMVAGlob::SetFrameStyle( sig, 1.2 );

      // normalise both signal and background
      TMVAGlob::NormalizeHists( sig, bgd );

      // finally plot and overlay
      Float_t sc = 1.1;
      if (countPad==2) sc = 1.3;
      sig->SetMaximum( TMath::Max( sig->GetMaximum(), bgd->GetMaximum() )*sc );
      sig->Draw( "hist" );

      bgd->Draw("histsame");
      sig->GetXaxis()->SetTitle( title );
      sig->GetYaxis()->SetTitleOffset( 1.30 );
      sig->GetYaxis()->SetTitle("Events");

      // Draw legend
      if (countPad==2){
         TLegend *legend= new TLegend( cPad->GetLeftMargin(), 
                                       1-cPad->GetTopMargin()-.15, 
                                       cPad->GetLeftMargin()+.4, 
                                       1-cPad->GetTopMargin() );
         legend->SetFillStyle(1);
         legend->AddEntry(sig,"Signal","F");
         legend->AddEntry(bgd,"Background","F");
         legend->Draw("same");
         legend->SetBorderSize(1);
         legend->SetMargin( 0.3 );
      } 

      // redraw axes
      sig->Draw("sameaxis");

      // save canvas to file
      if (countPad%noPadPerCanv==0) {
         TString fname = Form( "plots/%s_c%i", outfname[type].Data(), countCanvas );
         TMVAGlob::plot_logo();
         TMVAGlob::imgconv( canv, fname );
      }

   }

   return;
}
