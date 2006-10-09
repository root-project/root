void TMVAGui( const char * fName = "TMVA.root" ) 
{
   // filename not used at present ... 
   
   gROOT->Reset();
   //   gStyle->SetScreenFactor(2); // if you have a large screen, select 1,2 or 1.4

   // create the control bar
   cbar = new TControlBar( "vertical", "Plotting Scripts", 0, 0 );

   const char* buttonType = "button";

   // configure buttons
   cbar->AddButton( "Input Variables",
                    Form(".x variables.C(\"%s\")",fName),
                    "Plots all input variables (macro variables.C)",
                    buttonType );

   cbar->AddButton( "Decorrelated Variables",
                    Form(".x decorrelated_variables.C(\"%s\")",fName),    
                    "Plots all decorrelated input variables (macro decorrelated_variables.C)",
                    buttonType );

   cbar->AddButton( "Variable Correlations (summary)",
                    Form(".x correlations.C(\"%s\")",fName),
                    "Plots signal and background correlation summaries for all input variables (macro correlations.C)", 
                    buttonType );

   cbar->AddButton( "Variable Correlations (scatter profiles)",
                    Form(".x correlationscatters.C\(0,\"%s\")",fName), 
                    "Plots signal and background correlation profiles between all input variables (macro correlationscatters.C)",
                    buttonType );

   cbar->AddButton( "   Decorrelated-Variable Correlations (scatter profiles)   ",
                    Form(".x correlationscatters.C\(1,\"%s\")",fName), 
                    "Plots signal and background correlation profiles between all decorrelated input variables (macro correlationscatters.C(1))",
                    buttonType );

   cbar->AddButton( "Output MVA Variables",
                    Form(".x mvas.C(\"%s\")",fName),
                    "Plots the output variable of each method (macro mvas.C)",
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

   cbar->AddButton( "Quit",   ".q", "Quit", buttonType );

   // set the style 
   cbar->SetTextColor("black");

   // there seems to be a bug in ROOT: font jumps back to default after pressing on >2 different buttons
   // cbar->SetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   
   // draw
   cbar->Show();   

   gROOT->SaveContext();
}
