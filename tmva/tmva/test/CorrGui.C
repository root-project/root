///////////
//New Gui for easier plotting of scatter corelations
// L. Ancu 04/04/07
////////////
#include <iostream>

#include "TControlBar.h"
#include "tmvaglob.C"

static TControlBar* CorrGui_Global__cbar = 0;

void CorrGui(  TString fin = "TMVA.root", TString dirName = "InputVariables_Id", TString title = "TMVA Input Variable",
               Bool_t isRegression = kFALSE )
{
   // Use this script in order to run the various individual macros
   // that plot the output of TMVA (e.g. running TMVAnalysis.C),
   // stored in the file "TMVA.root"
   // for further documentation, look in the individual macros

   cout << "--- Open CorrGui for input file: " << fin << " and type: " << dirName << endl;

   // destroy all open cavases
   TMVAGlob::DestroyCanvases(); 
   
   TString extension = dirName;
   extension.ReplaceAll( "InputVariables", ""  );

   // create the control bar
   TControlBar* cbar = new TControlBar( "vertical", title, 50, 50 );
   CorrGui_Global__cbar = cbar;

   const char* buttonType = "button";
   const char* scriptpath = "./"; 

   // configure buttons      
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   TDirectory* dir = (TDirectory*)gDirectory->Get( dirName );
   if (!dir) {
      cout << "Could not locate directory '" << dirName << "' in file: " << fin << endl;
      return;
   }
   dir->cd();

   // how many variables  are in the directory?
   Int_t noVar = TMVAGlob::GetNumberOfInputVariables(dir);
   cout << "found number of variables='" << noVar<< endl;
   TString Var[noVar]; 

   TIter next(dir->GetListOfKeys());
   Int_t it=0;

   TKey *key;
   while ((key = (TKey*)next())) {

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (cl->InheritsFrom("TH1")) {
         TH1 *sig = (TH1*)key->ReadObj();
         TString hname = sig->GetName();
         // check for all signal histograms
         if (hname.Contains("__Signal") || (hname.Contains("__Regression") && !hname.Contains("__Regression_target"))) { // found a new signal plot
            hname.ReplaceAll(extension,"");
            hname.ReplaceAll("__Signal","");
            hname.ReplaceAll("__Regression","");
            Var[it] = hname;
            ++it;	
         }
      }
   }
   cout << "found histos for "<< it <<" variables='" << endl;
  
   for (Int_t ic=0;ic<it;ic++) {    
      cbar->AddButton( (Var[ic].Contains("_target") ? 
                        Form( "      Target: %s      ", Var[ic].ReplaceAll("_target","").Data()) : 
                        Form( "      Variable: %s      ", Var[ic].Data())),
                       Form( ".x %s/correlationscatters.C\(\"%s\",\"%s\",\"%s\",\"%s\",%i)", 
                             scriptpath, fin.Data(), Var[ic].Data(), dirName.Data(), title.Data(), (Int_t)isRegression ),
                       buttonType );
   }
      
   // *** problems with this button below ROOT 5.19 ***
#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
   cbar->AddButton( "Close", "CorrGui_DeleteTBar()", "Close this control bar", "button" );
#endif
   // **********************************************

   // set the style 
   cbar->SetTextColor("blue");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();

   gROOT->SaveContext();

}

void CorrGui_DeleteTBar()
{
   TMVAGlob::DestroyCanvases(); 

   delete CorrGui_Global__cbar;
   CorrGui_Global__cbar = 0;
}

