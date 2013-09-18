#include <iostream>
#include <vector>

#include "TROOT.h"
#include "TControlBar.h"
#include "tmvaglob.C"

// some global lists
static TList*               TMVAGui_keyContent;
static std::vector<TString> TMVAGui_inactiveButtons;

TList* GetKeyList( const TString& pattern )
{
   TList* list = new TList();

   TIter next( TMVAGui_keyContent );
   TKey* key(0);
   while ((key = (TKey*)next())) {         
      if (TString(key->GetName()).Contains( pattern )) { list->Add( new TObjString( key->GetName() ) ); }
   }
   return list;
}

// utility function
void ActionButton( TControlBar* cbar, 
                   const TString& title, const TString& macro, const TString& comment, 
                   const TString& buttonType, TString requiredKey = "" ) 
{
   cbar->AddButton( title, macro, comment, buttonType );

   // search    
   if (requiredKey != "") {
      Bool_t found = kFALSE;
      TIter next( TMVAGui_keyContent );
      TKey* key(0);
      while ((key = (TKey*)next())) {         
         if (TString(key->GetName()).Contains( requiredKey )) { found = kTRUE; break; }
      }
      if (!found) TMVAGui_inactiveButtons.push_back( title );
   }
}

// main GUI
void TMVARegGui( const char* fName = "TMVAReg.root" ) 
{   
   // Use this script in order to run the various individual macros
   // that plot the output of TMVA (e.g. running TMVARegression.C),
   // stored in the file "TMVA.Regroot"

   TString curMacroPath(gROOT->GetMacroPath());
   // uncomment next line for macros submitted to next root version
   gROOT->SetMacroPath(curMacroPath+":./:$ROOTSYS/tmva/test/:");

   // for the sourceforge version, including $ROOTSYS/tmva/test in the
   // macro path is a mistake, especially if "./" was not part of path
   // add ../macros to the path (comment out next line for the ROOT version of TMVA)
   // gROOT->SetMacroPath(curMacroPath+":../macros:");

   TString curIncludePath=gSystem->GetIncludePath();
   //std::cout <<"inc path="<<curIncludePath<<std::endl;
   TString newIncludePath=TString("-I../ ")+curIncludePath;
   gSystem->SetIncludePath(newIncludePath);
  
   cout << "--- Launch TMVA GUI to view input file: " << fName << endl;

   // init
   TMVAGui_inactiveButtons.clear();

   // check if file exist
   TFile* file = TFile::Open( fName );
   if (!file) {
      cout << "==> Abort TMVAGui, please verify filename" << endl;
      return;
   }
   // find all references   
   TMVAGui_keyContent = (TList*)file->GetListOfKeys()->Clone();

   // close file
   file->Close();

   TString defaultRequiredClassifier = "";

   //   gROOT->Reset();
   //   gStyle->SetScreenFactor(2); // if you have a large screen, select 1,2 or 1.4

   // create the control bar
   TControlBar* cbar = new TControlBar( "vertical", "TMVA Plotting Macros for Regression", 0, 0 );

   const TString buttonType( "button" );

   // configure buttons   
   Int_t ic = 1;

   // find all input variables types
   TList* keylist = GetKeyList( "InputVariables" );
   TListIter it( keylist );
   TObjString* str = 0;
   char ch = 'a';
   while (str = (TObjString*)it()) {
      TString tmp   = str->GetString();
      TString title = Form( "Input variables and target(s) '%s'-transformed (training sample)", 
                            tmp.ReplaceAll("InputVariables_","").Data() );
      if (tmp.Contains( "Id" )) title = "Input variables and target(s) (training sample)";
      ActionButton( cbar, 
                    Form( "    (%i%c) %s    ", ic, ch++, title.Data() ),
                    Form( ".x variables.C(\"%s\",\"%s\",\"%s\",kTRUE)", fName, str->GetString().Data(), title.Data() ),
                    Form( "Plots all '%s'-transformed input variables and target(s) (macro variables.C(...))", 
                          str->GetString().Data() ),
                    buttonType, str->GetString() );
   }      
   ic++;

   // correlation scatter plots 
   it.Reset(); ch = 'a';
   while (str = (TObjString*)it()) {
      TString tmp   = str->GetString();
      TString title = Form( "Input variable correlations '%s'-transformed (scatter profiles)", 
                            tmp.ReplaceAll("InputVariables_","").Data() );
      if (tmp.Contains( "Id" )) title = "Input variable correlations (scatter profiles)";
      ActionButton( cbar, 
                    Form( "(%i%c) %s", ic, ch++, title.Data() ),
                    Form( ".x CorrGui.C(\"%s\",\"%s\",\"%s\",kTRUE)", fName, str->GetString().Data(), title.Data() ),
                    Form( "Plots all correlation profiles between '%s'-transformed input variables (macro CorrGui.C(...))", 
                          str->GetString().Data() ),
                    buttonType, str->GetString() );
   }      
   
   // coefficients
   ActionButton( cbar,  
                 Form( "(%i) Input Variable Linear Correlation Coefficients", ++ic ),
                 Form( ".x correlations.C(\"%s\",kTRUE)", fName ),
                 "Plots signal and background correlation summaries for all input variables (macro correlations.C)", 
                 buttonType );

   ActionButton( cbar,  
                 Form( "(%ia) Regression Output Deviation versus Target (test sample)", ++ic ),
                 Form( ".x deviations.C(\"%s\",0,kTRUE)", fName ),
                 "Plots the deviation between regression output and target versus target on test data (macro deviations.C(...,0))",
                 buttonType, defaultRequiredClassifier );

   ActionButton( cbar,  
                 Form( "(%ib) Regression Output Deviation versus Target (training sample)", ic ),
                 Form( ".x deviations.C(\"%s\",3,kTRUE)", fName ),
                 "Plots the deviation between regression output and target versus target on test data (macro deviations.C(...,0))",
                 buttonType, defaultRequiredClassifier );

   ActionButton( cbar,  
                 Form( "(%ic) Regression Output Deviation versus Input Variables (test sample)", ic ),
                 Form( ".x deviations.C(\"%s\",0,kFALSE)", fName ),
                 "Plots the deviation between regression output and target versus target on test data (macro deviations.C(...,0))",
                 buttonType, defaultRequiredClassifier );

   ActionButton( cbar,  
                 Form( "   (%id) Regression Output Deviation versus Input Variables (training sample)   ", ic ),
                 Form( ".x deviations.C(\"%s\",3,kFALSE)", fName ),
                 "Plots the deviation between regression output and target versus target on test data (macro deviations.C(...,0))",
                 buttonType, defaultRequiredClassifier );

   ActionButton( cbar,  
                 Form( "(%i) Summary of Average Regression Deviations ", ++ic ),
                 Form( ".x regression_averagedevs.C(\"%s\")", fName ),
                 "Plot Summary of average deviations: MVAvalue - target (macro regression_averagedevs.C)",
                 buttonType );

   ActionButton( cbar,  
                 Form( "(%ia) Network Architecture", ++ic ),
                 Form( ".x network.C(\"%s\")", fName ), 
                 "Plots the MLP weights (macro network.C)",
                 buttonType, "MLP" );

   ActionButton( cbar,  
                 Form( "(%ib) Network Convergence Test", ic ),
                 Form( ".x annconvergencetest.C(\"%s\")", fName ), 
                 "Plots error estimator versus training epoch for training and test samples (macro annconvergencetest.C)",
                 buttonType, "MLP" );

   ActionButton( cbar,  
                 Form( "(%i) Plot Foams", ++ic ),
                 ".x PlotFoams.C(\"weights/TMVARegression_PDEFoam.weights_foams.root\")",
                 "Plot Foams (macro PlotFoams.C)",
                 buttonType, "PDEFoam" );

   ActionButton( cbar,  
                 Form( "(%i) Regression Trees (BDT)", ++ic ),
                 Form( ".x BDT_Reg.C+(\"%s\")", fName ),
                 "Plots the Regression Trees trained by BDT algorithms (macro BDT_Reg.C(itree,...))",
                 buttonType, "BDT" );

   ActionButton( cbar,  
                 Form( "(%i) Regression Tree Control Plots (BDT)", ++ic ),
                 Form( ".x BDTControlPlots.C(\"%s\")", fName ),
                 "Plots to monitor boosting and pruning of regression trees (macro BDTControlPlots.C)",
                 buttonType, "BDT" );

   cbar->AddSeparator();

   cbar->AddButton( Form( "(%i) Quit", ++ic ),   ".q", "Quit", buttonType );

   // set the style 
   cbar->SetTextColor("black");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();

   // indicate inactive buttons
   for (UInt_t i=0; i<TMVAGui_inactiveButtons.size(); i++) cbar->SetButtonState( TMVAGui_inactiveButtons[i], 3 );
   if (TMVAGui_inactiveButtons.size() > 0) {
      cout << "=== Note: inactive buttons indicate that the corresponding methods were not trained ===" << endl;
   }

   gROOT->SaveContext();
}
