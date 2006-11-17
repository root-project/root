#include <iostream>

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
   TControlBar * cbar = new TControlBar( "vertical", "Plotting Scripts", 0, 0 );

   const char* buttonType = "button";

   // configure buttons
   cbar->AddButton( "Input Variables",
                    Form(".x variables.C(\"%s\",0)",fName),
                    "Plots all input variables (macro variables.C)",
                    buttonType );
   
   cbar->AddButton( "Decorrelated Variables",
                    Form(".x variables.C(\"%s\",1)",fName),    
                    "Plots all decorrelated input variables (macro variables.C)",
                    buttonType );

   cbar->AddButton( "PCA-transformed Variables",
                    Form(".x variables.C(\"%s\",2)",fName),    
                    "Plots all PCA-transformed input variables (macro variables.C)",
                    buttonType );

   cbar->AddButton( "Variable Correlations (scatter profiles)",
                    Form(".x correlationscatters.C\(\"%s\",0)",fName), 
                    "Plots signal and background correlation profiles between all input variables (macro correlationscatters.C)",
                    buttonType );

   cbar->AddButton( "   Decorrelated-Variable Correlations (scatter profiles)   ",
                    Form(".x correlationscatters.C\(\"%s\",1)",fName), 
                    "Plots signal and background correlation profiles between all decorrelated input variables (macro correlationscatters.C(1))",
                    buttonType );

   cbar->AddButton( "   PCA-transformed Variable Correlations (scatter profiles)   ",
                    Form(".x correlationscatters.C\(\"%s\",2)",fName), 
                    "Plots signal and background correlation profiles between all PCA-transformed input variables (macro correlationscatters.C(2))",
                    buttonType );

   cbar->AddButton( "Variable Correlations (summary)",
                    Form(".x correlations.C(\"%s\")",fName),
                    "Plots signal and background correlation summaries for all input variables (macro correlations.C)", 
                    buttonType );

   cbar->AddButton( "Output MVA Variables",
                    Form(".x mvas.C(\"%s\")",fName),
                    "Plots the output variable of each method (macro mvas.C)",
                    buttonType );

   cbar->AddButton( "Mu-transforms (summary)",
                    Form(".x mutransform.C(\"%s\")",fName),
                    "Plots the mu-transformed signal and background MVAs of each method (macro mutransform.C)",
                    buttonType );

   cbar->AddButton( "Background Rejection vs Signal Efficiencies",
                    Form(".x efficiencies.C(\"%s\")",fName),
                    "Plots background rejection vs signal efficiencies (macro efficiencies.C)",
                    buttonType );

   cbar->AddButton( "Likelihood Reference Distributiuons (if exist)",
                    Form(".x likelihoodrefs.C(\"%s\")",fName), 
                    "Plots to verify the likelihood reference distributions (macro likelihoodrefs.C)",
                    buttonType );

   cbar->AddButton( "Network Architecture (if exists)",
                    Form(".x network.C(\"%s\")",fName), 
                   "Plots the MLP weights (macro network.C)",
                    buttonType );

   cbar->AddButton( "Network Convergence Test (if exists)",
                    Form(".x annconvergencetest.C(\"%s\")",fName), 
                    "Plots error estimator versus training epoch for training and test samples (macro annconvergencetest.C)",
                    buttonType );

   cbar->AddButton( "Decision Tree (#1)",
                    Form(".x BDT.C",fName),
                    "Plots the Decision Tree (#1); to plot other trees (i) call macro BDT.C(i) from command line",
                    buttonType );


   cbar->AddButton( "Quit",   ".q", "Quit", buttonType );

   // set the style 
   cbar->SetTextColor("black");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();



   gROOT->SaveContext();
}
