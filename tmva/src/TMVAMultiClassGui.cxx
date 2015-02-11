#include "TMVA/TMVAMultiClassGui.h"
#include <iostream>
#include <vector>

#include "TList.h"
#include "TROOT.h"
#include "TKey.h"
#include "TString.h"
#include "TControlBar.h"
#include "TObjString.h"



// some global lists
static TList*               TMVAMultiClassGui_keyContent;
static std::vector<TString> TMVAMultiClassGui_inactiveButtons;

TList* TMVA::MultiClassGetKeyList( const TString& pattern )
{
   TList* list = new TList();

   TIter next( TMVAMultiClassGui_keyContent );
   TKey* key(0);
   while ((key = (TKey*)next())) {         
      if (TString(key->GetName()).Contains( pattern )) { list->Add( new TObjString( key->GetName() ) ); }
   }
   return list;
}

// utility function
void TMVA::MultiClassActionButton( TControlBar* cbar, 
                   const TString& title, const TString& macro, const TString& comment, 
                   const TString& buttonType, TString requiredKey ) 
{
   cbar->AddButton( title, macro, comment, buttonType );

   // search    
   if (requiredKey != "") {
      Bool_t found = kFALSE;
      TIter next( TMVAMultiClassGui_keyContent );
      TKey* key(0);
      while ((key = (TKey*)next())) {         
         if (TString(key->GetName()).Contains( requiredKey )) { found = kTRUE; break; }
      }
      if (!found) TMVAMultiClassGui_inactiveButtons.push_back( title );
   }
}

// main GUI
void TMVA::TMVAMultiClassGui( const char* fName ) 
{   
   // Use this script in order to run the various individual macros
   // that plot the output of TMVA (e.g. running TMVAClassification.cxx),
   // stored in the file "TMVA.root"

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
   std::cout <<"new include path="<<gSystem->GetIncludePath()<<std::endl;
  
   std::cout << "--- Launch TMVA GUI to view input file: " << fName << std::endl;

   // init
   TMVAMultiClassGui_inactiveButtons.clear();

   // check if file exist
   TFile* file = TFile::Open( fName );
   if (!file) {
      std::cout << "==> Abort TMVAMultiClassGui, please verify filename" << std::endl;
      return;
   }
   // find all references   
   TMVAMultiClassGui_keyContent = (TList*)file->GetListOfKeys()->Clone();

   //close file
   file->Close();

   TString defaultRequiredClassifier = "";

   //   gROOT->Reset();
   //   gStyle->SetScreenFactor(2); // if you have a large screen, select 1,2 or 1.4

   // create the control bar
   TControlBar* cbar = new TControlBar( "vertical", "TMVA Plotting Macros for Multiclass Classification", 0, 0 );

   const TString buttonType( "button" );

   // configure buttons   
   Int_t ic = 1;

   // find all input variables types
   TList* keylist = MultiClassGetKeyList( "InputVariables" );
   TListIter it( keylist );
   TObjString* str = 0;
   char ch = 'a';
   while ((str = (TObjString*)it())) {
      TString tmp   = str->GetString();
      TString title = Form( "Input variables '%s'-transformed (training sample)", 
                            tmp.ReplaceAll("InputVariables_","").Data() );
      if (tmp.Contains( "Id" )) title = "Input variables (training sample)";
      MultiClassActionButton( cbar, 
                    Form( "(%i%c) %s", ic, ch++, title.Data() ),
                    Form( "TMVA::variablesMultiClass(\"%s\",\"%s\",\"%s\")", fName, str->GetString().Data(), title.Data() ),
                    Form( "Plots all '%s'-transformed input variables (macro variablesMultiClass(...))", str->GetString().Data() ),
                    buttonType, str->GetString() );
   }      
   ic++;

   // correlation scatter plots 
   it.Reset(); ch = 'a';
   while ((str = (TObjString*)it())) {
      TString tmp   = str->GetString();
      TString title = Form( "Input variable correlations '%s'-transformed (scatter profiles)", 
                            tmp.ReplaceAll("InputVariables_","").Data() );
      if (tmp.Contains( "Id" )) title = "Input variable correlations (scatter profiles)";
      MultiClassActionButton( cbar, 
                    Form( "(%i%c) %s", ic, ch++, title.Data() ),
                    Form( "TMVA::CorrGuiMultiClass(\"%s\",\"%s\",\"%s\")", fName, str->GetString().Data(), title.Data() ),
                    Form( "Plots all correlation profiles between '%s'-transformed input variables (macro CorrGuiMultiClass(...))", 
                          str->GetString().Data() ),
                    buttonType, str->GetString() );
   }      
   
   TString title;
   // coefficients
   title =Form( "(%i) Input Variable Linear Correlation Coefficients", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::correlationsMultiClass(\"%s\")", fName ),
                 "Plots signal and background correlation summaries for all input variables (macro correlationsMultiClass.cxx)", 
                 buttonType );

   title =Form( "(%ia) Classifier Output Distributions (test sample)", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::mvasMulticlass(\"%s\",TMVA::kMVAType)", fName ),
                 "Plots the output of each classifier for the test data (macro mvas(...,0))",
                 buttonType, defaultRequiredClassifier );

   title =Form( "(%ib) Classifier Output Distributions (test and training samples superimposed)", ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::mvasMulticlass(\"%s\",TMVA::kCompareType)", fName ),
                 "Plots the output of each classifier for the test (histograms) and training (dots) data (macro mvas(...,3))",
                 buttonType, defaultRequiredClassifier );
   /*
     title = Form( "(%ic) Classifier Probability Distributions (test sample)", ic );
     MultiClassActionButton( cbar,  
     Form( "(%ic) Classifier Probability Distributions (test sample)", ic ),
     Form( "TMVA::mvas(\"%s\",TMVA::kProbaType)", fName ),
     "Plots the probability of each classifier for the test data (macro mvas(...,1))",
     buttonType, defaultRequiredClassifier );
     
     title =Form( "(%id) Classifier Rarity Distributions (test sample)", ic );
     MultiClassActionButton( cbar,  
     Form( "(%id) Classifier Rarity Distributions (test sample)", ic ),
     Form( "TMVA::mvas(\"%s\",TMVA::kRarityType)", fName ),
     "Plots the Rarity of each classifier for the test data (macro mvas(...,2)) - background distribution should be uniform",
     buttonType, defaultRequiredClassifier );
   
   
   title =Form( "(%ia) Classifier Cut Efficiencies", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::mvaeffs(\"%s\")", fName ),
                 "Plots signal and background efficiencies versus cut on classifier output (macro mvaeffs.cxx)",
                 buttonType, defaultRequiredClassifier );

   title = Form( "(%ib) Classifier Background Rejection vs Signal Efficiency (ROC curve)", ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::efficiencies(\"%s\")", fName ),
                 "Plots background rejection vs signal efficiencies (macro efficiencies.cxx) [\"ROC\" stands for \"Receiver Operation Characteristics\"]",
                 buttonType, defaultRequiredClassifier );
                 
   
   title = Form( "(%i) Parallel Coordinates (requires ROOT-version >= 5.17)", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::paracoor(\"%s\")", fName ),
                 "Plots parallel coordinates for classifiers and input variables (macro paracoor.cxx, requires ROOT >= 5.17)",
                 buttonType, defaultRequiredClassifier );

   // parallel coordinates only exist since ROOT 5.17
   #if ROOT_VERSION_CODE < ROOT_VERSION(5,17,0)
   TMVAMultiClassGui_inactiveButtons.push_back( title );
   #endif
   
   
   title =Form( "(%i) PDFs of Classifiers (requires \"CreateMVAPdfs\" option set)", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::probas(\"%s\")", fName ),
                 "Plots the PDFs of the classifier output distributions for signal and background - if requested (macro probas.cxx)",
                 buttonType, defaultRequiredClassifier );

   title = Form( "(%i) Likelihood Reference Distributiuons", ++ic);
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::likelihoodrefs(\"%s\")", fName ), 
                 "Plots to verify the likelihood reference distributions (macro likelihoodrefs.cxx)",
                 buttonType, "Likelihood" );
   */
   
   title = Form( "(%ia) Network Architecture (MLP)", ++ic );
   TString call = Form( "TMVA::network(\"%s\")", fName );
   MultiClassActionButton( cbar,  
                 title,
                 call, 
                 "Plots the MLP weights (macro network.cxx)",
                 buttonType, "MLP" );

   title = Form( "(%ib) Network Convergence Test (MLP)", ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::annconvergencetest(\"%s\")", fName ), 
                 "Plots error estimator versus training epoch for training and test samples (macro annconvergencetest.cxx)",
                 buttonType, "MLP" );

   title = Form( "(%i) Decision Trees (BDT)", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::BDT(\"%s\")", fName ),
                 "Plots the Decision Trees trained by BDT algorithms (macro BDT(itree,...))",
                 buttonType, "BDT" );

   /*
     title = Form( "(%i) Decision Tree Control Plots (BDT)", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::BDTControlPlots(\"%s\")", fName ),
                 "Plots to monitor boosting and pruning of decision trees (macro BDTControlPlots.cxx)",
                 buttonType, "BDT" );

   */
   title = Form( "(%i) Plot Foams (PDEFoam)", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 "TMVA::PlotFoams(\"weights/TMVAMulticlass_PDEFoam.weights_foams.root\")",
                 "Plot Foams (macro PlotFoams.cxx)",
                 buttonType, "PDEFoam" );
   /*
   title = Form( "(%i) General Boost Control Plots", ++ic );
   MultiClassActionButton( cbar,  
                 title,
                 Form( "TMVA::BoostControlPlots(\"%s\")", fName ),
                 "Plots to monitor boosting of general classifiers (macro BoostControlPlots.cxx)",
                 buttonType, "Boost" );
   */
   cbar->AddSeparator();

   cbar->AddButton( Form( "(%i) Quit", ++ic ),   ".q", "Quit", buttonType );

   // set the style 
   cbar->SetTextColor("black");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();

   // indicate inactive buttons
   for (UInt_t i=0; i<TMVAMultiClassGui_inactiveButtons.size(); i++) cbar->SetButtonState( TMVAMultiClassGui_inactiveButtons[i], 3 );
   if (TMVAMultiClassGui_inactiveButtons.size() > 0) {
      std::cout << "=== Note: inactive buttons indicate that the corresponding classifiers were not trained ===" << std::endl;
   }

   gROOT->SaveContext();
}
