///////////
//New Gui for easier plotting of scatter corelations
// L. Ancu 04/04/07
////////////
#include <iostream>

#include "TControlBar.h"
#include "tmvaglob.C"

static TControlBar* CorrGui_Global__cbar = 0;

void CorrGui(  TString fin = "TMVA.root",  TMVAGlob::TypeOfPlot type = TMVAGlob::kNormal ) 
{
   // Use this script in order to run the various individual macros
   // that plot the output of TMVA (e.g. running TMVAnalysis.C),
   // stored in the file "TMVA.root"
   // for further documentation, look in the individual macros

   cout << "--- Open CorrGui for input file: " << fin << " and type: " << type << endl;

   // destroy all open cavases
   TMVAGlob::DestroyCanvases(); 
   
   //   gROOT->Reset();
   //   gStyle->SetScreenFactor(2); // if you have a large screen, select 1,2 or 1.4

   // create the control bar
   TControlBar* cbar = new TControlBar( "vertical", "Plotting correlations", 50, 50 );
   CorrGui_Global__cbar = cbar;

   const char* buttonType = "button";
   const char* scriptpath = "./"; 

   // configure buttons   

   const TString directories[TMVAGlob::kNumOfMethods] = { "InputVariables_NoTransform",
                                                          "InputVariables_DecorrTransform",
                                                          "InputVariables_PCATransform",
                                                          "InputVariables_GaussDecorr"
   };

   
   const TString titles[TMVAGlob::kNumOfMethods] = { "TMVA Input Variable",
                                                     "Decorrelated TMVA Input Variables",
                                                     "Principal Component Transformed TMVA Input Variables" ,
                                                     "Gaussianized and Decorrelated TMVA Input Variable"
   };
   
   const TString extensions[TMVAGlob::kNumOfMethods] = { "_NoTransform",
                                                         "_DecorrTransform",
                                                         "_PCATransform", 
                                                         "_GaussDecorr" 
   };
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   TDirectory* dir = (TDirectory*)gDirectory->Get( directories[type] );
   if (!dir) {
      cout << "Could not locate directory '" << directories[type] << "' in file: " << fin << endl;
      return;
   }
   dir->cd();

   // how many variables  are in the directory?
   Int_t noVar = ((dir->GetListOfKeys())->GetEntries()) / 2;
   const TString Var[noVar]; 
 
   TIter next(dir->GetListOfKeys());
   Int_t it=0;

   TKey *key;
   while ((key = (TKey*)next())) {

      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1")) continue;
      TH1 *sig = (TH1*)key->ReadObj();
      TString hname= sig->GetName();

      // check for all signal histograms
      if (hname.Contains("__S")){ // found a new signal plot
         hname.ReplaceAll(extensions[type],"");
         hname.ReplaceAll("__S","");
         Var[it]+=hname;
        
         ++it;	
      }
   }


   for (Int_t ic=0;ic<it;ic++) {    
      cbar->AddButton( Form( "Variable %s",Var[ic].Data()),
                       Form( ".x %s/correlationscatters.C\(\"%s\",\"%s\",%i)", 
                             scriptpath, fin.Data(), Var[ic].Data(), type ),
                       Form( "Draws all scatter plots for variable \"%s\"",Var[ic].Data() ),
                       buttonType );
   }
      
   // *** problems with this button in ROOT 5.19 ***
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

