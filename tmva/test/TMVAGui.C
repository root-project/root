#include <iostream>

#include "TROOT.h"
#include "TControlBar.h"
#include "tmvaglob.C"

void TMVAGui( const char* fName = "TMVA.root" ) 
{
   // Use this script in order to run the various individual macros
   // that plot the output of TMVA (e.g. running TMVAnalysis.C),
   // stored in the file "TMVA.root"
   // for further documentation, look in the individual macros

   cout << "--- Launch TMVA GUI to view input file: " << fName << endl;

   //   gROOT->Reset();
   //   gStyle->SetScreenFactor(2); // if you have a large screen, select 1,2 or 1.4

   // create the control bar
   TControlBar* cbar = new TControlBar( "vertical", "TMVA Plotting Macros", 0, 0 );

   const char* buttonType = "button";

   // configure buttons   
   Int_t ic = 0;
   cbar->AddButton( Form( "(%ia) Input Variables", ++ic),
                    Form( ".x variables.C(\"%s\",0)", fName ),
                    "Plots all input variables (macro variables.C)",
                    buttonType );
   
   cbar->AddButton( Form( "(%ib) [ Decorrelated Input Variables ]", ic ),
                    Form( ".x variables.C(\"%s\",1)", fName ),
                    "Plots all decorrelated input variables (macro variables.C(1))",
                    buttonType );

   cbar->AddButton( Form( "(%ic) [ PCA-transformed Input Variables ]", ic ),
                    Form( ".x variables.C(\"%s\",2)", fName ),    
                    "Plots all PCA-transformed input variables (macro variables.C(2))",
                    buttonType );

   cbar->AddButton( Form( "(%ia) Input Variable Correlations (scatter profiles)", ++ic ),
                    Form( ".x CorrGui.C\(\"%s\",0)", fName ), 
                    "Plots signal and background correlation profiles between input variables (macro CorrGui.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ib) [ Decorrelated Input Variable Correlations (scatter profiles) ]", ic ),
                    Form( ".x CorrGui.C\(\"%s\",1)", fName ), 
                    "Plots signal and background correlation profiles between decorrelated input variables (macro CorrGui.C(1))",
                    buttonType );

   cbar->AddButton( Form( "(%ic) [ PCA-transformed Input Variable Correlations (scatter profiles) ]", ic ),
                    Form( ".x CorrGui.C\(\"%s\",2)", fName ), 
                    "Plots signal and background correlation profiles between PCA-transformed input variables (macro CorrGui.C(2))",
                    buttonType );

   cbar->AddButton( Form( "(%i) Input Variable Correlation Coefficients", ++ic ),
                    Form( ".x correlations.C(\"%s\")", fName ),
                    "Plots signal and background correlation summaries for all input variables (macro correlations.C)", 
                    buttonType );

   cbar->AddButton( Form( "(%ia) Classifier Output Distributions", ++ic ),
                    Form( ".x mvas.C(\"%s\",0)", fName ),
                    "Plots the output of each classifier for the test data (macro mvas.C(...,0))",
                    buttonType );

   cbar->AddButton( Form( "(%ib) Classifier Output Distributions for Training and Test Samples", ic ),
                    Form( ".x mvas.C(\"%s\",3)", fName ),
                    "Plots the rarity of each classifier for the test data (macro mvas.C(...,3))",
                    buttonType );

   cbar->AddButton( Form( "(%ic) Classifier Probability Distributions", ic ),
                    Form( ".x mvas.C(\"%s\",1)", fName ),
                    "Plots the probability of each classifier for the test data (macro mvas.C(...,1))",
                    buttonType );

   cbar->AddButton( Form( "(%id) Classifier Rarity Distributions", ic ),
                    Form( ".x mvas.C(\"%s\",2)", fName ),
                    "Plots the rarity of each classifier for the test data (macro mvas.C(...,2))",
                    buttonType );

   cbar->AddButton( Form( "(%ia) Classifier Cut Efficiencies", ++ic ),
                    Form( ".x mvaeffs.C(\"%s\")", fName ),
                    "Plots signal and background efficiencies versus cut on classifier output (macro mvaeffs.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ib) Classifier Background Rejection vs Signal Efficiency (ROC curve)", ic ),
                    Form( ".x efficiencies.C(\"%s\")", fName ),
                    "Plots background rejection vs signal efficiencies (macro efficiencies.C)",
                    buttonType );

   cbar->AddButton( Form( "(%i) [ Likelihood Reference Distributiuons ]", ++ic),
                    Form( ".x likelihoodrefs.C(\"%s\")", fName ), 
                    "Plots to verify the likelihood reference distributions (macro likelihoodrefs.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ia) [ Network Architecture ]", ++ic ),
                    Form( ".x network.C(\"%s\")", fName ), 
                   "Plots the MLP weights (macro network.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ib) [ Network Convergence Test ]", ic ),
                    Form( ".x annconvergencetest.C(\"%s\")", fName ), 
                    "Plots error estimator versus training epoch for training and test samples (macro annconvergencetest.C)",
                    buttonType );

   cbar->AddButton( Form( "(%i) [ Decision Tree (#1) ]", ++ic ),
                    Form( ".x BDT.C", fName ),
                    "Plots the Decision Tree (#1); to plot other trees (i) call macro BDT.C(i) from command line",
                    buttonType );

   cbar->AddButton( Form( "(%i) PDFs of Classifiers", ++ic ),
                    Form( ".x probas.C(\"%s\")", fName ),
                    "Plots the Fit of the Methods outputs; to plot other trees (i) call macro from command line (macro probas.C(i))",
                    buttonType );

   cbar->AddButton( Form( "(%i) [ Rule Ensemble Importance Plots ]", ++ic ),
                    Form( ".x rulevis.C(\"%s\",0)", fName ),
                    "Plots all input variables with rule ensemble weights, including linear terms (macro rulevis.C)",
                    buttonType );

   cbar->AddButton( Form( "(%i) Quit", ++ic ),   ".q", "Quit", buttonType );

   // set the style 
   cbar->SetTextColor("black");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();

   gROOT->SaveContext();
}
