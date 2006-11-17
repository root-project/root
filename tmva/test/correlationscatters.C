#include "tmvaglob.C"

// this macro plots the correlations (as scatter plots) of
// the various input variable combinations used in TMVA (e.g. running
// TMVAnalysis.C).  Signal and Background are plotted separately

const TString extensions[TMVAGlob::kNumOfMethods] = { "",
                                                      "_decorr",
                                                      "_PCA" };

// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void correlationscatters( TString fin = "TMVA.root", TMVAGlob::TypeOfPlot type = TMVAGlob::kNormal, 
                          bool useTMVAStyle = kTRUE )
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   TString dirName = "CorrelationPlots" + extensions[type];
  
   TDirectory* dir = (TDirectory*)gDirectory->Get( dirName );
   if (dir==0) {
      cout << "Could not locate directory: " << dirName << " in file " << fin << endl;
      return;
   }
   dir->cd();

   TListIter keyIt(dir->GetListOfKeys());
   Int_t noPlots = 0;
   TKey * key = 0;
   // how many plots are in the directory?
   Int_t noPlots = ((dir->GetListOfKeys())->GetEntries()) / 2;
   cout << "--- Found: " << noPlots << " plots in directory: " << dirName << endl;

   // define Canvas layout here!
   // default setting
   Int_t xPad;  // no of plots in x
   Int_t yPad;  // no of plots in y
   Int_t width; // size of canvas
   Int_t height;
   switch (noPlots) {
   case 1:
      xPad = 1; yPad = 1; width = 500; height = width; break;
   case 2:
      xPad = 2; yPad = 1; width = 600; height = 0.7*width; break;
   case 3:
      xPad = 3; yPad = 1; width = 800; height = 0.5*width; break;
   case 4:
      xPad = 2; yPad = 2; width = 600; height = width; break;
   default:
      xPad = 3; yPad = 2; width = 800; height = 0.7*width; break;
   }
   Int_t noPad = xPad * yPad ;   

   // this defines how many canvases we need
   const Int_t noCanvas = 1 + (Int_t)(noPlots/noPad);
   TCanvas **c = new TCanvas*[noCanvas];
   for (Int_t ic=0; ic<noCanvas; ic++) c[ic] = 0;

   cout << "--- Found: " << noPlots << " plots; "
        << "will produce: " << noCanvas << " canvas" << endl;

   // counter variables
   Int_t countCanvas = 0;
   Int_t countPad    = 1;

   // loop over all objects in "input_variables" directory
   TString thename[2] = { "_sig", "_bgd" };
   for (UInt_t itype = 0; itype < 2; itype++) {

      TIter next(gDirectory->GetListOfKeys());
      TKey* key = 0;

      while ((key = (TKey*)next())) {

         if (key->GetCycle() != 1) continue;

         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(key->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TH1 *scat = (TH1*)key->ReadObj();
         TString hname= scat->GetName();

         // check for all signal histograms
         if (hname.EndsWith( thename[itype] + extensions[type] ) && 
             hname.BeginsWith( "scat_" )) { // found a new signal plot

            cout << hname << endl;

            // create new canvas
            if ((c[countCanvas]==NULL) || (countPad>noPad)) {
               cout << "--- Book canvas no: " << countCanvas << endl;
               char cn[20];
               sprintf( cn, "canvas%d", countCanvas+1 );
               c[countCanvas] = new TCanvas( cn, "Correlation Profiles", 
                                             countCanvas*50+200, countCanvas*20, width, height ); 
               // style
               c[countCanvas]->SetBorderMode(0);
               c[countCanvas]->SetFillColor(0);

               c[countCanvas]->Divide(xPad,yPad);
               countPad = 1;
            }       

            // save canvas to file
            c[countCanvas]->cd(countPad);
            countPad++;

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
      
            // finally plot and overlay       
            Float_t sc = 1.1;
            if (countPad==2) sc = 1.3;
            scat->SetMarkerColor(  4);
            scat->Draw();      
            prof->SetMarkerColor( 2 );
            prof->SetMarkerSize( 0.2 );
            prof->SetLineColor( 2 );
            prof->SetLineWidth( 1 );
            prof->SetFillStyle( 3002 );
            prof->SetFillColor( 46 );
            prof->Draw("samee1");

            // redraw axes
            scat->Draw("sameaxis");
      
            // save canvas to file
            if (countPad > noPad) {
               c[countCanvas]->Update();

               TString fname = Form( "plots/correlationscatter_%s_c%i", extensions[type].Data(), countCanvas+1 );
               TMVAGlob::imgconv( c[countCanvas], &fname[0] );
               countCanvas++;
            }
         }
      }
   }
   if (countPad <= noPad) {
      c[countCanvas]->Update();
      TString fname = Form( "plots/correlationscatter_%s_c%i", extensions[type].Data(), countCanvas+1 );
      TMVAGlob::imgconv( c[countCanvas], &fname[0] );
   }

}
