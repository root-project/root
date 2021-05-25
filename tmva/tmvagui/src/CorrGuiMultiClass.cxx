#include "TMVA/CorrGuiMultiClass.h"
///////////
//New Gui for easier plotting of scatter corelations
// L. Ancu 04/04/07
////////////
#include <iostream>

#include "TControlBar.h"


//static TControlBar* CorrGuiMultiClass_Global__cbar = 0;

void TMVA::CorrGuiMultiClass(TString dataset,  TString fin , TString dirName , TString title ,
                             Bool_t isRegression )
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
   //   CorrGuiMultiClass_Global__cbar = cbar;

   const char* buttonType = "button";

   // configure buttons      
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  

   TDirectory* dir = (TDirectory*)file->GetDirectory(dataset.Data())->Get( dirName );
   if (!dir) {
      cout << "Could not locate directory '" << dirName << "' in file: " << fin << endl;
      return;
   }
   dir->cd();

   // how many variables  are in the directory?
   std::vector<TString> names(TMVAGlob::GetInputVariableNames(dir));
   cout << "found number of variables='" << names.end() - names.begin() << endl;

   std::vector<TString>::const_iterator iter = names.begin();
   for (; iter != names.end(); ++iter) {    
      cbar->AddButton( Form( "      Variable: %s      ", (*iter).Data()),
                       Form( "TMVA::correlationscattersMultiClass(\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%i)",
                             dataset.Data(), fin.Data(), (*iter).Data(), dirName.Data(), title.Data(), (Int_t)isRegression ),
                       buttonType );
   }

   // set the style 
   cbar->SetTextColor("blue");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();

   gROOT->SaveContext();

}

void TMVA::CorrGuiMultiClass_DeleteTBar()
{
   TMVAGlob::DestroyCanvases(); 

   //   delete CorrGuiMultiClass_Global__cbar;
   //   CorrGuiMultiClass_Global__cbar = 0;
}

