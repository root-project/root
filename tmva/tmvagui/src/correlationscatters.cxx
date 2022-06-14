#include "TMVA/correlationscatters.h"
#include "TMVA/Config.h"



// this macro plots the correlations (as scatter plots) of
// the various input variable combinations used in TMVA (e.g. running
// TMVAnalysis.C).  Signal and Background are plotted separately

// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void TMVA::correlationscatters(TString dataset, TString fin , TString var, 
                               TString dirName_, TString /*title */ ,
                               Bool_t isRegression ,
                               Bool_t useTMVAStyle  )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   TString extension = dirName_;
   extension.ReplaceAll( "InputVariables", "" );
   extension.ReplaceAll( " ", "" );
   if (extension == "") extension = "_Id"; // use 'Id' for 'idendtity transform'

   var.ReplaceAll( extension, "" );
   cout << "Called macro \"correlationscatters\" for variable: \"" << var 
        << "\", transformation type \"" << dirName_ 
        << "\" (extension: \"" << extension << "\")" << endl;

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   TString dirName = dirName_ + "/CorrelationPlots";
  
   // find out number of input variables   
   TDirectory* vardir = (TDirectory*)file->GetDirectory(dataset.Data())->Get("InputVariables_Id");
   if (!vardir) {
      cout << "ERROR: no such directory: \"InputVariables\"" << endl;
      return;
   }
   Int_t noVars = TMVAGlob::GetNumberOfInputVariables( vardir ); // subtraction of target(s) no longer necessary

   TDirectory* dir = (TDirectory*)file->GetDirectory(dataset.Data())->Get( dirName );
   if (dir==0) {
      cout << "No information about " << extension << " available in " << fin << endl;
      return;
   }
   dir->cd();

   TListIter keyIt(dir->GetListOfKeys());
   Int_t noPlots = noVars - 1;
   
   cout << "noPlots: " << noPlots << " --> noVars: " << noVars << endl;
   if (noVars != Int_t(noVars)) {
      cout << "*** Warning: problem in inferred number of variables ... not an integer *** " << endl;
   }

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
      xPad = 3; yPad = 1; width = 800; height = 0.4*width; break;
   case 4:
      xPad = 2; yPad = 2; width = 600; height = width; break;
   default:
      xPad = 3; yPad = 2; width = 800; height = 0.55*width; break;
   }
   Int_t noPadPerCanv = xPad * yPad ;   

   // counter variables
   Int_t countCanvas = 0;

   // loop over all objects in "input_variables" directory
   TString thename[2] = { "_Signal", "_Background" };
   if (isRegression) thename[0] = "_Regression";
   for (UInt_t itype = 0; itype < 2; itype++) {

      TIter next(gDirectory->GetListOfKeys());
      TKey   * key  = 0;
      TCanvas* canv = 0;

      Int_t countPad    = 0;
   
      while ( (key = (TKey*)next()) ) {

         if (key->GetCycle() != 1) continue;

         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(key->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *scat = (TH1*)key->ReadObj();
         TString hname = scat->GetName();

         // check for all signal histograms
         if (! (hname.EndsWith( thename[itype] + extension ) && 
                hname.Contains( TString("_") + var + "_" ) && hname.BeginsWith("scat_")) ) {
            scat->Delete();
            continue; 
         }

         // found a new signal plot
            
         // create new canvas
         if (countPad%noPadPerCanv==0) {
            ++countCanvas;
            TString ext = extension; ext.Remove( 0, 1 );
            canv = new TCanvas( Form("canvas%d", countCanvas), 
                                Form("Correlation profiles for '%s'-transformed %s variables", 
                                     ext.Data(), (isRegression ? "" : (itype==0) ? "signal" : "background")),
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
            cout << "ERROR!!! couldn't find background histo for" << hname << endl;
            //exit(1);
            return;
         }
         // this is set but not stored during plot creation in MVA_Factory
         TMVAGlob::SetSignalAndBackgroundStyle( scat, prof );

         // chop off "signal" 
         TMVAGlob::SetFrameStyle( scat, 1.2 );

         // normalise both signal and background
         scat->Scale( 1.0/scat->GetSumOfWeights() );

         // finally plot and overlay       
         scat->SetMarkerColor(  4);
         scat->Draw("col");      
         prof->SetMarkerColor( gConfig().fVariablePlotting.fUsePaperStyle ? 1 : 2  );
         prof->SetMarkerSize( 0.2 );
         prof->SetLineColor( gConfig().fVariablePlotting.fUsePaperStyle ? 1 : 2 );
         prof->SetLineWidth( gConfig().fVariablePlotting.fUsePaperStyle ? 2 : 1 );
         prof->SetFillStyle( 3002 );
         prof->SetFillColor( 46 );
         prof->Draw("samee1");
         // redraw axes
         scat->Draw("sameaxis");

         // save canvas to file
         if (countPad%noPadPerCanv==0) {
            canv->Update();

            TString fname = Form( "%s/plots/correlationscatter_%s_%s_c%i",dataset.Data(),var.Data(), extension.Data(), countCanvas );
            TMVAGlob::plot_logo();
            TMVAGlob::imgconv( canv, fname );
         }
      }
      if (countPad%noPadPerCanv!=0) {
         canv->Update();

         TString fname = Form( "%s/plots/correlationscatter_%s_%s_c%i",dataset.Data(),var.Data(), extension.Data(), countCanvas );
         TMVAGlob::plot_logo();
         TMVAGlob::imgconv( canv, fname );
      }
   }
}
