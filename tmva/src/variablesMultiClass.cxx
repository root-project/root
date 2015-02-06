#include "TMVA/variablesMultiClass.h"


// this macro plots the distributions of the different input variables
// used in TMVA (e.g. running TMVAnalysis.C).  Signal and Background are overlayed.

// input: - Input file (result from TMVA),
//        - normal/decorrelated/PCA
//        - use of TMVA plotting TStyle
void TMVA::variablesMultiClass( TString fin , TString dirName , TString title,
                                Bool_t /* isRegression */, Bool_t useTMVAStyle )
{
   TString outfname = dirName;
   TString tmp = dirName;
   tmp.ReplaceAll("InputVariables_","");
   outfname.ToLower(); outfname.ReplaceAll( "input", ""  );

   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );

   // obtain shorter histogram title 
   TString htitle = title; 
   htitle.ReplaceAll("variables ","variable");
   htitle.ReplaceAll("and target(s)","");
   htitle.ReplaceAll("(training sample)","");

   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );

   TDirectory* dir = (TDirectory*)file->Get( dirName );
   if (dir==0) {
      cout << "No information about " << title << " available in directory " << dirName << " of file " << fin << endl;
      return;
   }
   dir->cd();

   // how many plots are in the directory?
   Int_t noPlots = TMVAGlob::GetNumberOfInputVariables( dir );

   // define Canvas layout here!
   // default setting
   Int_t xPad;  // no of plots in x
   Int_t yPad;  // no of plots in y
   Int_t width; // size of canvas
   Int_t height;
   switch (noPlots) {
   case 1:
      xPad = 1; yPad = 1; width = 550; height = 0.90*width; break;
   case 2:
      xPad = 2; yPad = 1; width = 600; height = 0.50*width; break;
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
   TCanvas* canv = 0;
   Bool_t   createNewFig = kFALSE;
   TIter next(dir->GetListOfKeys());
    
   std::vector<TString> varnames(TMVAGlob::GetInputVariableNames(dir));
   std::vector<TString> classnames(TMVAGlob::GetClassNames(dir));


   std::vector<TString>::iterator variter = varnames.begin();
   std::vector<TString>::iterator classiter = classnames.begin();

   /*
   std::vector<TString>::const_iterator variter = varnames.begin();
   std::cout << "Available variables:" << std::endl;
   while(variter != varnames.end()){
      std::cout << *variter << std::endl;
      variter++;
   }
   
   std::vector<TString>::const_iterator classiter = classnames.begin();
   std::cout << "Available classes:" << std::endl;
   while(classiter != classnames.end()){
      std::cout << *classiter << std::endl;
      classiter++;
   }
   */
   
   variter = varnames.begin();
   for(; variter!=varnames.end(); ++variter){
      
      //create new canvas
      if (countPad%noPadPerCanv==0) {
         ++countCanvas;
         canv = new TCanvas( Form("canvas%d", countCanvas), title,
                             countCanvas*50+50, countCanvas*20, width, height );
         canv->Divide(xPad,yPad);
         canv->Draw();
      }
      TPad* cPad = (TPad*)canv->cd(countPad++%noPadPerCanv+1);
      classiter = classnames.begin();
      
      TObjArray hists;
      for(; classiter!=classnames.end(); ++classiter){
         //assemble histogram names
         TString hname(*variter + "__" + *classiter + "_" + tmp);
         TH1 *hist = (TH1*)dir->Get(hname);
         //cout << "Looking for histgram " << hname << endl;
         if (hist == NULL) {
            cout << "ERROR!!! couldn't find " << *variter << " histogram for class " << *classiter << endl;
            //exit(1);
            return;
         }
         hists.Add(hist);
      }
      
      // this is set but not stored during plot creation in MVA_Factory  
      //TMVAGlob::SetSignalAndBackgroundStyle(((TH1*)hists[0]), ((TH1*)hists[1]));            
      TMVAGlob::SetMultiClassStyle( &hists ); 
      
      ((TH1*)hists.First())->SetTitle( TString( htitle ) + ": " + *variter );

      TMVAGlob::SetFrameStyle( ((TH1*)hists.First()), 1.2 );
      
      // normalise all histograms and find maximum
      Float_t histmax = -1;
      for(Int_t i=0; i<hists.GetEntriesFast(); ++i){
         TMVAGlob::NormalizeHist((TH1*)hists[i] );
         if(((TH1*)hists[i])->GetMaximum() > histmax)
            histmax = ((TH1*)hists[i])->GetMaximum();
      }
      
      // finally plot and overlay
      Float_t sc = 1.1;
      if (countPad == 1) sc = 1.3;
      ((TH1*)hists.First())->SetMaximum( histmax*sc );
      
      ((TH1*)hists.First())->Draw( "hist" );
      cPad->SetLeftMargin( 0.17 );
      ((TH1*)hists.First())->GetYaxis()->SetTitleOffset( 1.70 );
      
      for(Int_t i=1; i<hists.GetEntriesFast(); ++i){

         ((TH1*)hists[i])->Draw("histsame");
         TString ytit = TString("(1/N) ") + ((TH1*)hists[i])->GetYaxis()->GetTitle();
         ((TH1*)hists[i])->GetYaxis()->SetTitle( ytit ); // histograms are normalised
      
      }
           
      // Draw legend
      if (countPad == 1) {
         TLegend *legend= new TLegend( cPad->GetLeftMargin(), 
                                       1-cPad->GetTopMargin()-.15, 
                                       cPad->GetLeftMargin()+.4, 
                                       1-cPad->GetTopMargin() );
         legend->SetFillStyle(1);
         
         classiter = classnames.begin();

         for(Int_t i=0; i<hists.GetEntriesFast(); ++i, ++classiter){
            legend->AddEntry(((TH1*)hists[i]),*classiter,"F");
         }

         legend->SetBorderSize(1);
         legend->SetMargin( 0.3 );
         legend->Draw("same");
      } 

      // redraw axes
      ((TH1*)hists.First())->Draw("sameaxis");

      
      // text for overflows
      Int_t    nbin = ((TH1*)hists.First())->GetNbinsX();
      Double_t dxu  = ((TH1*)hists.First())->GetBinWidth(0);
      Double_t dxo  = ((TH1*)hists.First())->GetBinWidth(nbin+1);
      TString uoflow = "";
           
      classiter = classnames.begin();
      for(Int_t i=0; i<hists.GetEntriesFast(); ++i, ++classiter){
         if(((TH1*)hists[i])->GetBinContent(0)!=0 || ((TH1*)hists[i])->GetBinContent(nbin+1)!=0){
            uoflow += *classiter;
            uoflow += Form( " U/O-flow:  %.1f / %.1f %%", 
                            ((TH1*)hists[i])->GetBinContent(0)*dxu*100, ((TH1*)hists[i])->GetBinContent(nbin+1)*dxo*100);
         }
      }
      
      TText* t = new TText( 0.98, 0.14, uoflow );
      t->SetNDC();
      t->SetTextSize( 0.040 );
      t->SetTextAngle( 90 );
      t->AppendPad();    
      
      
      // save canvas to file
      if (countPad%noPadPerCanv==0) {
         TString fname = Form( "plots/%s_c%i", outfname.Data(), countCanvas );
         TMVAGlob::plot_logo();
         TMVAGlob::imgconv( canv, fname );
         createNewFig = kFALSE;
      }
      else {
         createNewFig = kTRUE;
      }
   } 
   
   if (createNewFig) {
      TString fname = Form( "plots/%s_c%i", outfname.Data(), countCanvas );
      TMVAGlob::plot_logo();
      TMVAGlob::imgconv( canv, fname );
      createNewFig = kFALSE;
   }

   return;
}
