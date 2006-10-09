/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVApplication                                                     *
 *                                                                                *
 * This exectutable provides a simple example on how to use the trained MVAs      *
 * within a C++ analysis module                                                   *
 *                                                                                *
 * ------------------------------------------------------------------------------ *
 * see also the alternative (slightly faster) way to retrieve the MVA values in   *
 * examples/TMVApplicationAlternative.cxx                                         *
 * ------------------------------------------------------------------------------ *
 **********************************************************************************/

// ---------------------------------------------------------------
// choose MVA methods to be trained + tested
Bool_t Use_Cuts           = 1;
Bool_t Use_Likelihood     = 1;
Bool_t Use_LikelihoodD    = 1;
Bool_t Use_PDERS          = 1;
Bool_t Use_PDERSD         = 1;
Bool_t Use_HMatrix        = 1;
Bool_t Use_Fisher         = 1;
Bool_t Use_MLP            = 1;
Bool_t Use_CFMlpANN       = 0;
Bool_t Use_TMlpANN        = 0;
Bool_t Use_BDT            = 1;
Bool_t Use_RuleFit        = 1;
// ---------------------------------------------------------------

void TMVApplication() 
{
   cout << endl;
   cout << "==> start TMVApplication" << endl;

   //
   // create the Reader object
   //
   TMVA::Reader *reader = new TMVA::Reader();    

   // create a set of variables and declare them to the reader
   // - the variable names must corresponds in name and type to 
   // those given in the weight file(s) that you use
   Float_t var1, var2, var3, var4;
   reader->AddVariable( "var1", &var1 );
   reader->AddVariable( "var2", &var2 );
   reader->AddVariable( "var3", &var3 );
   reader->AddVariable( "var4", &var4 );

   //
   // book the MVA methods
   //
   string dir    = "../macros/weights/";
   string prefix = "MVAnalysis";

   if (Use_Cuts)        reader->BookMVA( "Cuts method",        dir + prefix + "_Cuts.weights.txt"     );
   if (Use_Likelihood)  reader->BookMVA( "Likelihood method",  dir + prefix + "_Likelihood.weights.txt"  );
   if (Use_LikelihoodD) reader->BookMVA( "LikelihoodD method", dir + prefix + "_LikelihoodD.weights.txt" );
   if (Use_PDERS)       reader->BookMVA( "PDERS method",       dir + prefix + "_PDERS.weights.txt"  );
   if (Use_PDERSD)      reader->BookMVA( "PDERSD method",      dir + prefix + "_PDERSD.weights.txt"  );
   if (Use_HMatrix)     reader->BookMVA( "HMatrix method",     dir + prefix + "_HMatrix.weights.txt"  );
   if (Use_Fisher)      reader->BookMVA( "Fisher method",      dir + prefix + "_Fisher.weights.txt"   );
   if (Use_MLP)         reader->BookMVA( "MLP method",         dir + prefix + "_MLP.weights.txt" );
   if (Use_CFMlpANN)    reader->BookMVA( "CFMlpANN method",    dir + prefix + "_CFMlpANN.weights.txt" );
   if (Use_TMlpANN)     reader->BookMVA( "TMlpANN method",     dir + prefix + "_TMlpANN.weights.txt"  );
   if (Use_BDT)         reader->BookMVA( "BDT method",         dir + prefix + "_BDT.weights.txt"  );
   if (Use_RuleFit)     reader->BookMVA( "RuleFit method",     dir + prefix + "_RuleFit.weights.txt"  );
  
   //
   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //   
   TFile *input      = new TFile("../examples/data/toy_sigbkg.root");
   TTree *signal     = (TTree*)input->Get("TreeS");
   TTree *background = (TTree*)input->Get("TreeB");  

   // 
   // book output histograms
   UInt_t nbin = 100;
	TH1F *histLk, *histLkD, *histPD, *histPDD, *histHm, *histFi, *histNn, *histNnC, *histNnT, *histBdt, *histRf;
   if (Use_Likelihood)  histLk  = new TH1F( "MVA_Likelihood",  "MVA_Likelihood",  nbin,  0, 1 );
   if (Use_LikelihoodD) histLkD = new TH1F( "MVA_LikelihoodD", "MVA_LikelihoodD", nbin,  0, 1 );
   if (Use_PDERS)       histPD  = new TH1F( "MVA_PDERS",       "MVA_PDERS",       nbin,  0, 1 );
   if (Use_PDERSD)      histPDD = new TH1F( "MVA_PDERSD",      "MVA_PDERSD",      nbin,  0, 1 );
   if (Use_HMatrix)     histHm  = new TH1F( "MVA_HMatrix",     "MVA_HMatrix",     nbin, -0.95, 1.55 );
   if (Use_Fisher)      histFi  = new TH1F( "MVA_Fisher",      "MVA_Fisher",      nbin, -4, 4 );
   if (Use_MLP)         histNn  = new TH1F( "MVA_MLP",         "MVA_MLP",         nbin, -0.25, 1.5 );
   if (Use_CFMlpANN)    histNnC = new TH1F( "MVA_CFMlpANN",    "MVA_CFMlpANN",    nbin,  0, 1 );
   if (Use_TMlpANN)     histNnT = new TH1F( "MVA_TMlpANN",     "MVA_TMlpANN",     nbin, -1, 1 );
   if (Use_BDT)         histBdt = new TH1F( "MVA_BDT",         "MVA_BDT",         nbin, -0.4, 0.6 );
   if (Use_RuleFit)     histRf  = new TH1F( "MVA_RuleFit",     "MVA_RuleFit",     nbin, -1.3, 1.3 );

   //
   // prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = signal;
   theTree->SetBranchAddress( "var1", &var1 );
   theTree->SetBranchAddress( "var2", &var2 );
   theTree->SetBranchAddress( "var3", &var3 );
   theTree->SetBranchAddress( "var4", &var4 );

   // efficiency calculator for cut method
   int    nSelCuts = 0;
   double effS     = 0.7;

   cout << "--- processing: " << theTree->GetEntries() << " events" << endl;
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      theTree->GetEntry(ievt);
    
      if (ievt%1000 == 0)
         cout << "--- ... processing event: " << ievt << endl;

      // 
      // return the MVA
      // 
      if (Use_Cuts) {
         // give the desired signal efficienciy
         Bool_t passed = reader->EvaluateMVA( "Cuts method", effS );
         if (passed) nSelCuts++;
      }

      //
      // fill histograms
      //
      if (Use_Likelihood ) histLk ->Fill( reader->EvaluateMVA( "Likelihood method"  ) );
      if (Use_LikelihoodD) histLkD->Fill( reader->EvaluateMVA( "LikelihoodD method" ) );
      if (Use_PDERS      ) histPD ->Fill( reader->EvaluateMVA( "PDERS method"       ) );
      if (Use_PDERSD     ) histPDD->Fill( reader->EvaluateMVA( "PDERSD method"      ) );
      if (Use_HMatrix    ) histHm ->Fill( reader->EvaluateMVA( "HMatrix method"     ) );
      if (Use_Fisher     ) histFi ->Fill( reader->EvaluateMVA( "Fisher method"      ) );
      if (Use_MLP        ) histNn ->Fill( reader->EvaluateMVA( "MLP method"         ) );
      if (Use_CFMlpANN   ) histNnC->Fill( reader->EvaluateMVA( "CFMlpANN method"    ) );
      if (Use_TMlpANN    ) histNnT->Fill( reader->EvaluateMVA( "TMlpANN method"     ) );
      if (Use_BDT        ) histBdt->Fill( reader->EvaluateMVA( "BDT method"         ) );        
      if (Use_RuleFit    ) histRf ->Fill( reader->EvaluateMVA( "RuleFit method"     ) );        
   }
   cout << "--- end of event loop" << endl;
   // get elapsed time
   if (Use_Cuts) cout << "--- efficiency for cut method: " << double(nSelCuts)/theTree->GetEntries()
                      << " (for a required signal efficiency of " << effS << ")" << endl;

   //
   // write histograms
   //
   TFile *target  = new TFile( "TMVApp.root","RECREATE" );
   if (Use_Likelihood  ) histLk ->Write();
   if (Use_LikelihoodD ) histLkD->Write();
   if (Use_PDERS       ) histPD ->Write();
   if (Use_PDERSD      ) histPDD->Write();
   if (Use_HMatrix     ) histHm ->Write();
   if (Use_Fisher      ) histFi ->Write();
   if (Use_MLP         ) histNn ->Write();
   if (Use_CFMlpANN    ) histNnC->Write();
   if (Use_TMlpANN     ) histNnT->Write();
   if (Use_BDT         ) histBdt->Write();
   if (Use_RuleFit     ) histRf ->Write();
   target->Close();

   cout << "--- created root file: \"TMVApp.root\" containing the MVA output histograms" << endl;
  
   delete reader;
    
   cout << "==> TMVApplication is done!" << endl << endl;
} 
