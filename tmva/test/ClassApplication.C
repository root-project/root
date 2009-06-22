/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: ClassApplication                                                   *
 *                                                                                *
 * Test suit for comparison of Reader and standalone class outputs                *
 **********************************************************************************/

#include <vector>

void ClassApplication( TString myMethodList = "Fisher" ) 
{
   cout << endl;
   cout << "==> start ClassApplication" << endl;
   const int Nmvas = 16;

   const char* bulkname[Nmvas] = { "MLP","MLPBFGS","Fisher","FisherG","Likelihood","LikelihoodD","LikelihoodPCA","LD","HMatrix","FDA_MT","FDA_MC","FDA_GA","BDT","BDTD","BDTG","BDTB"};

   bool iuse[Nmvas] = { Nmvas*kFALSE };

   // interpret input list
   if (myMethodList != "") {
      TList* mlist = TMVA::gTools().ParseFormatLine( myMethodList, " :," );
      for (int imva=0; imva<Nmvas; imva++) if (mlist->FindObject( bulkname[imva] )) iuse[imva] = kTRUE;
      delete mlist;
   }

   // create a set of variables and declare them to the reader
   // - the variable names must corresponds in name and type to 
   // those given in the weight file(s) that you use
   std::vector<std::string> inputVars;
   inputVars.push_back( "var1+var2" );
   inputVars.push_back( "var1-var2" );
   inputVars.push_back( "var3" );
   inputVars.push_back( "var4" );
      
   // preload standalone class(es)
   string dir    = "weights/";
   string prefix = "TMVAClassification";

   for (int imva=0; imva<Nmvas; imva++) {
      if (iuse[imva]) {
         TString cfile = dir + prefix + "_" + bulkname[imva] + ".class.C++";

         cout << "=== Macro        : Loading class  file: " << cfile << endl;         

         // load the classifier's standalone class
         gROOT->LoadMacro( cfile );
      }
   }
   cout << "=== Macro        : Classifier class loading successfully terminated" << endl;

   // define classes
   IClassifierReader* classReader[Nmvas] = { Nmvas*0 };

   // ... and create them (and some histograms for the output)
   int nbin = 100;
   TH1* hist[Nmvas];

   for (int imva=0; imva<Nmvas; imva++) {
      if (iuse[imva]) {
         cout << "=== Macro        : Testing " << bulkname[imva] << endl;
         if (bulkname[imva] == "Likelihood"   ) {
            classReader[imva] = new ReadLikelihood   ( inputVars );            
            hist[imva] = new TH1F( "MVA_Likelihood",    "MVA_Likelihood",    nbin,  0, 1 );
         }
         if (bulkname[imva] == "LikelihoodD"  ) {
            classReader[imva] = new ReadLikelihoodD  ( inputVars );
            hist[imva] = new TH1F( "MVA_LikelihoodD",   "MVA_LikelihoodD",   nbin,  0, 1 );
         }
         if (bulkname[imva] == "LikelihoodPCA") {
            classReader[imva] = new ReadLikelihoodPCA( inputVars );
            hist[imva] = new TH1F( "MVA_LikelihoodPCA", "MVA_LikelihoodPCA", nbin,  0, 1 );
         }
         if (bulkname[imva] == "LikelihoodMIX") {
            classReader[imva] = new ReadLikelihoodMIX( inputVars );
            hist[imva] = new TH1F( "MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin,  0, 1 );
         }
         if (bulkname[imva] == "HMatrix"      ) {
            classReader[imva] = new ReadHMatrix      ( inputVars );
            hist[imva] = new TH1F( "MVA_HMatrix",       "MVA_HMatrix",       nbin, -0.95, 1.55 );
         }
         if (bulkname[imva] == "Fisher"       ) {
            classReader[imva] = new ReadFisher       ( inputVars );
            hist[imva] = new TH1F( "MVA_Fisher",        "MVA_Fisher",        nbin, -4, 4 );
         }
         if (bulkname[imva] == "FisherG"       ) {
            classReader[imva] = new ReadFisherG       ( inputVars );
            hist[imva] = new TH1F( "MVA_FisherG",        "MVA_FisherG",        nbin, -4, 4 );
         }
	 if (bulkname[imva] == "LD"   ) {
            classReader[imva] = new ReadLD   ( inputVars );            
            hist[imva] = new TH1F( "MVA_LD",    "MVA_LD",    nbin,  -1., 1 );
         }	 
         if (bulkname[imva] == "FDA_MT"       ) {
            classReader[imva] = new ReadFDA_MT       ( inputVars );
            hist[imva] = new TH1F( "MVA_FDA_MT",        "MVA_FDA_MT",        nbin, -2.0, 3.0 );
         }
         if (bulkname[imva] == "FDA_MC"       ) {
            classReader[imva] = new ReadFDA_MC       ( inputVars );
            hist[imva] = new TH1F( "MVA_FDA_MC",        "MVA_FDA_MC",        nbin, -2.0, 3.0 );
         }
         if (bulkname[imva] == "FDA_GA"       ) {
            classReader[imva] = new ReadFDA_GA       ( inputVars );
            hist[imva] = new TH1F( "MVA_FDA_GA",        "MVA_FDA_GA",        nbin, -2.0, 3.0 );
         }
         if (bulkname[imva] == "MLP"          ) {
            classReader[imva] = new ReadMLP          ( inputVars );
            hist[imva] = new TH1F( "MVA_MLP",           "MVA_MLP",           nbin, -1.2, 1.2 );
         }
         if (bulkname[imva] == "MLPBFGS"          ) {
            classReader[imva] = new ReadMLPBFGS          ( inputVars );
            hist[imva] = new TH1F( "MVA_MLPBFGS",           "MVA_MLPBFGS",           nbin, -1.5, 1.5 );
         }
         if (bulkname[imva] == "BDT"          ) {
            classReader[imva] = new ReadBDT          ( inputVars );
            hist[imva] = new TH1F( "MVA_BDT",           "MVA_BDT",           nbin, -1, 1 );
         }
         if (bulkname[imva] == "BDTD"          ) {
            classReader[imva] = new ReadBDTD          ( inputVars );
            hist[imva] = new TH1F( "MVA_BDTD",           "MVA_BDTD",         nbin, -1, 1 );
         }
         if (bulkname[imva] == "BDTG"          ) {
            classReader[imva] = new ReadBDTG          ( inputVars );
            hist[imva] = new TH1F( "MVA_BDTG",           "MVA_BDTG",         nbin, -1, 1 );
         }
         if (bulkname[imva] == "BDTB"          ) {
            classReader[imva] = new ReadBDTB          ( inputVars );
            hist[imva] = new TH1F( "MVA_BDTB",           "MVA_BDTB",         nbin, -1, 1 );
         }
      }
   }
   cout << "=== Macro        : Class creation was successful" << endl;

   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //   
   TFile *input(0);
   if (!gSystem->AccessPathName("./tmva_example.root")) {
      // first we try to find tmva_example.root in the local directory
      cout << "=== Macro        : Accessing ./tmva_example.root" << endl;
      input = TFile::Open("tmva_example.root");
   } 
   
   if (!input) {
      cout << "ERROR: could not open data file" << endl;
      exit(1);
   }

   //
   // prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = (TTree*)input->Get("TreeS");
   cout << "=== Macro        : Loop over signal sample" << endl;

   // the references to the variables
   float var1, var2, var3, var4;
   float userVar1, userVar2;
   theTree->SetBranchAddress( "var1", &userVar1 );
   theTree->SetBranchAddress( "var2", &userVar2 );
   theTree->SetBranchAddress( "var3", &var3 );
   theTree->SetBranchAddress( "var4", &var4 );

   cout << "=== Macro        : Processing total of " << theTree->GetEntries() << " events ... " << endl;

   std::vector<double>* inputVec = new std::vector<double>( 4 );
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0) cout << "=== Macro        : ... processing event: " << ievt << endl;

      theTree->GetEntry(ievt);

      var1 = userVar1 + userVar2;
      var2 = userVar1 - userVar2;

      (*inputVec)[0] = var1;
      (*inputVec)[1] = var2;
      (*inputVec)[2] = var3;
      (*inputVec)[3] = var4;
      
      // loop over all booked classifiers
      for (int imva=0; imva<Nmvas; imva++) {

         if (iuse[imva]) {
            
            // retrive the classifier responses            
            double retval = classReader[imva]->GetMvaValue( *inputVec );
            hist[imva]->Fill( retval, 1.0 );
         }
      }
   }
   
   cout << "=== Macro        : Event loop done! " << endl;

   TFile *target  = new TFile( "ClassApp.root","RECREATE" );
   for (int imva=0; imva<Nmvas; imva++) {
      if (iuse[imva]) {
         hist[imva]->Write();
      }
   }
   cout << "=== Macro        : Created target file: " << target->GetName() << endl;
   target->Close();

   delete target;
   delete inputVec;
    
   cout << "==> ClassApplication is done!" << endl << endl;
} 
