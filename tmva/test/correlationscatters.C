#include "tmvaglob.C"

// this macro plots the correlations (as scatter plots) of
// the various input variable combinations used in TMVA (e.g. running
// TMVAnalysis.C).  Signal and Background are plotted separately

// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void correlationscatters( TString fin = "TMVA.root", TString var= "var3", TMVAGlob::TypeOfPlot type = TMVAGlob::kNormal, bool useTMVAStyle = kTRUE )
{

   const TString directories[TMVAGlob::kNumOfMethods] = { "InputVariables_NoTransform",
                                                          "InputVariables_DecorrTransform",
                                                          "InputVariables_PCATransform" }; 

   const TString titles[3] = { "TMVA Input Variable",
                               "Decorrelated TMVA Input Variables",
                               "Principal Component Transformed TMVA Input Variables" };
  
   const TString extensions[TMVAGlob::kNumOfMethods] = { "_NoTransform",
                                                         "_DecorrTransform",
                                                         "_PCATransform" };

   cout << "Called macro \"correlationscatters\" with type: " << type << endl;

   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   TString dirName = directories[type] + "/CorrelationPlots";
  
   TDirectory* dir = (TDirectory*)gDirectory->Get( dirName );
   if (dir==0) {
      cout << "No information about " << titles[type] << " available in " << fin << endl;
      return;
   }
   dir->cd();

   TListIter keyIt(dir->GetListOfKeys());
   Int_t noPlots = 0;
   TKey* key = 0;
   // how many plots are in the directory?
   Int_t    noPlots = (TMVAGlob::UsePaperStyle) ? 1 : ((dir->GetListOfKeys())->GetEntries());
   Double_t noVars  = noPlots == 1 ? 0 : (1 + TMath::Sqrt(1.0 + 2.0*noPlots))/2.0;
   cout << "noPlots: " << noPlots << " --> noVars: " << noVars << endl;
   if (noVars != Int_t(noVars)) {
      cout << "*** Warning: problem in inferred number of variables ... not an integer *** " << endl;
   }
   noPlots = noVars;

   // define Canvas layout here!
   // default setting
   Int_t xPad;  // no of plots in x
   Int_t yPad;  // no of plots in y
   Int_t width; // size of canvas
   Int_t height;
   switch (noPlots) {
   case 1:
      xPad = 1; yPad = 1; width = 400; height = width; break;
   case 2:
      xPad = 2; yPad = 1; width = 700; height = 0.55*width; break;
   case 3:
      xPad = 3; yPad = 1; width = 800; height = 0.5*width; break;
   case 4:
      xPad = 2; yPad = 2; width = 600; height = width; break;
   default:
      xPad = 3; yPad = 2; width = 800; height = 0.55*width; break;
   }
   Int_t noPadPerCanv = xPad * yPad ;   

   // counter variables
   Int_t countCanvas = 0;

   // loop over all objects in "input_variables" directory
   TString thename[2] = { "_sig", "_bgd" };
   for (UInt_t itype = 0; itype < 2; itype++) {

      TIter next(gDirectory->GetListOfKeys());
      TKey   * key  = 0;
      TCanvas* canv = 0;

      Int_t countPad    = 0;
   
      while ((key = (TKey*)next())) {

         if (key->GetCycle() != 1) continue;

         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(key->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *scat = (TH1*)key->ReadObj();
         TString hname= scat->GetName();
         
         // check for all signal histograms
         if (! (hname.EndsWith( thename[itype] + extensions[type] ) && 
                hname.Contains( "_"+var+"_" ) && hname.BeginsWith("scat_")) ) continue; 
                  
         // found a new signal plot
            
         // create new canvas
         if (countPad%noPadPerCanv==0) {
            ++countCanvas;
            canv = new TCanvas( Form("canvas%d", countCanvas), 
                                Form("Correlation Profiles for %s", (itype==0) ? "Signal" : "Background"),
                                countCanvas*50+200, countCanvas*20, width, height ); 
            canv->Divide(xPad,yPad);
         }

         if (!canv) continue;

         canv->cd(countPad++%noPadPerCanv+1);

         // find the corredponding backgrouns histo
         TString bgname = hname;
         bgname.ReplaceAll("scat_","prof_");
         TH1 *prof = (TH1*)gDirectory->Get(bgname);
         if (prof == NULL) {
            cout << "ERROR!!! couldn't find backgroung histo for" << hname << endl;
            exit;
         }
         // this is set but not stored during plot creation in MVA_Factory
         TMVAGlob::SetSignalAndBackgroundStyle( scat, prof );

         // chop off "signal" 
         TMVAGlob::SetFrameStyle( scat, 1.2 );

         // normalise both signal and background
         scat->Scale( 1.0/scat->GetSumOfWeights() );

         // finally plot and overlay       
         Float_t sc = 1.1;
         if (countPad==2) sc = 1.3;
         scat->SetMarkerColor(  4);
         scat->Draw();      
         prof->SetMarkerColor( TMVAGlob::UsePaperStyle ? 1 : 2  );
         prof->SetMarkerSize( 0.2 );
         prof->SetLineColor( TMVAGlob::UsePaperStyle ? 1 : 2 );
         prof->SetLineWidth( TMVAGlob::UsePaperStyle ? 2 : 1 );
         prof->SetFillStyle( 3002 );
         prof->SetFillColor( 46 );
         prof->Draw("samee1");
         // redraw axes
         scat->Draw("sameaxis");

         // save canvas to file
         if (countPad%noPadPerCanv==0) {
            canv->Update();

            TString fname = Form( "plots/correlationscatter_%s_%s_c%i",var.Data(), extensions[type].Data(), countCanvas );
            TMVAGlob::plot_logo();
            TMVAGlob::imgconv( canv, fname );
         }
      }
      if (countPad%noPadPerCanv!=0) {
         canv->Update();

         TString fname = Form( "plots/correlationscatter_%s_%s_c%i",var.Data(), extensions[type].Data(), countCanvas );
         TMVAGlob::plot_logo();
         TMVAGlob::imgconv( canv, fname );
      }
   }
}
