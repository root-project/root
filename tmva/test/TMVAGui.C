#include <iostream>

#include "TROOT.h"
#include "TControlBar.h"

void TMVAGui( const char* fName = "TMVA.root" ) 
{
   // Use this script in order to run the various individual macros
   // that plot the output of TMVA (e.g. running TMVAnalysis.C),
   // stored in the file "TMVA.root"
   // for further documentation, look in the individual macros

   cout << "--- Open TMVAGui for input file: " << fName << endl;

   //   gROOT->Reset();
   //   gStyle->SetScreenFactor(2); // if you have a large screen, select 1,2 or 1.4

   // create the control bar
   TControlBar* cbar = new TControlBar( "vertical", "TMVA Plotting Macros", 0, 0 );

   const char* buttonType = "button";
   const char* scriptpath = "../macros";

   // configure buttons   
   Int_t ic = 0;
   cbar->AddButton( Form( "(%ia) Input Variables", ++ic),
                    Form( ".x %s/variables.C(\"%s\",0)", scriptpath, fName ),
                    "Plots all input variables (macro variables.C)",
                    buttonType );
   
   cbar->AddButton( Form( "(%ib) [ Decorrelated Input Variables ]", ic ),
                    Form( ".x %s/variables.C(\"%s\",1)", scriptpath, fName ),
                    "Plots all decorrelated input variables (macro variables.C(1))",
                    buttonType );

   cbar->AddButton( Form( "(%ic) [ PCA-transformed Input Variables ]", ic ),
                    Form( ".x %s/variables.C(\"%s\",2)", scriptpath, fName ),    
                    "Plots all PCA-transformed input variables (macro variables.C(2))",
                    buttonType );

   cbar->AddButton( Form( "(%ia) Input Variable Correlations (scatter profiles)", ++ic ),
                    Form( ".x %s/CorrGui.C\(\"%s\",0)", scriptpath, fName ), 
                    "Plots signal and background correlation profiles between input variables (macro CorrGui.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ib) [ Decorrelated Input Variable Correlations (scatter profiles) ]", ic ),
                    Form( ".x %s/CorrGui.C\(\"%s\",1)", scriptpath, fName ), 
                    "Plots signal and background correlation profiles between decorrelated input variables (macro CorrGui.C(1))",
                    buttonType );

   cbar->AddButton( Form( "(%ic) [ PCA-transformed Input Variable Correlations (scatter profiles) ]", ic ),
                    Form( ".x %s/CorrGui.C\(\"%s\",2)", scriptpath, fName ), 
                    "Plots signal and background correlation profiles between PCA-transformed input variables (macro CorrGui.C(2))",
                    buttonType );

   cbar->AddButton( Form( "(%i) Input Variable Correlation Coefficients", ++ic ),
                    Form( ".x %s/correlations.C(\"%s\")", scriptpath, fName ),
                    "Plots signal and background correlation summaries for all input variables (macro correlations.C)", 
                    buttonType );

   cbar->AddButton( Form( "(%ia) Classifier Output Distributions", ++ic ),
                    Form( ".x %s/mvas.C(\"%s\",0)", scriptpath, fName ),
                    "Plots the output of each classifier for the test data (macro mvas.C(...,0))",
                    buttonType );

   cbar->AddButton( Form( "(%ib) Classifier Probability Distributions", ic ),
                    Form( ".x %s/mvas.C(\"%s\",1)", scriptpath, fName ),
                    "Plots the probability of each classifier for the test data (macro mvas.C(...,1))",
                    buttonType );

   //    cbar->AddButton( Form( "(%i) Mu-transforms (summary)", ++ic ),
   //                     Form( ".x %s/mutransform.C(\"%s\")", scriptpath, fName ),
   //                     "Plots the mu-transformed signal and background MVAs of each method (macro mutransform.C)",
   //                     buttonType );

   cbar->AddButton( Form( "(%ia) Classifier Cut Efficiencies", ++ic ),
                    Form( ".x %s/mvaeffs.C(\"%s\")", scriptpath, fName ),
                    "Plots signal and background efficiencies versus cut on classifier output (macro mvaeffs.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ib) Classifier Background Rejection vs Signal Efficiency (ROC curve)", ic ),
                    Form( ".x %s/efficiencies.C(\"%s\")", scriptpath, fName ),
                    "Plots background rejection vs signal efficiencies (macro efficiencies.C)",
                    buttonType );

   cbar->AddButton( Form( "(%i) [ Likelihood Reference Distributiuons ]", ++ic),
                    Form( ".x %s/likelihoodrefs.C(\"%s\")", scriptpath, fName ), 
                    "Plots to verify the likelihood reference distributions (macro likelihoodrefs.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ia) [ Network Architecture ]", ++ic ),
                    Form( ".x %s/network.C(\"%s\")", scriptpath, fName ), 
                   "Plots the MLP weights (macro network.C)",
                    buttonType );

   cbar->AddButton( Form( "(%ib) [ Network Convergence Test ]", ic ),
                    Form( ".x %s/annconvergencetest.C(\"%s\")", scriptpath, fName ), 
                    "Plots error estimator versus training epoch for training and test samples (macro annconvergencetest.C)",
                    buttonType );

   cbar->AddButton( Form( "(%i) [ Decision Tree (#1) ]", ++ic ),
                    Form( ".x %s/BDT.C", scriptpath, fName ),
                    "Plots the Decision Tree (#1); to plot other trees (i) call macro BDT.C(i) from command line",
                    buttonType );

   cbar->AddButton( Form( "(%i) PDFs of Classifiers", ++ic ),
                    Form( ".x %s/probas.C(\"%s\")", scriptpath, fName ),
                    "Plots the Fit of the Methods outputs; to plot other trees (i) call macro from command line (macro probas.C(i))",
                    buttonType );

   cbar->AddButton( Form( "(%i) [ Rule Ensemble Importance Plots ]", ++ic ),
                    Form( ".x %s/rulevis.C(\"%s\",0)", scriptpath, fName ),
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
