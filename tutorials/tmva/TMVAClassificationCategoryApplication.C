/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides a simple example on how to use the trained classifiers
/// (with categories) within an analysis module
/// - Project   : TMVA - a Root-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Exectuable: TMVAClassificationCategoryApplication
///
/// \macro_output
/// \macro_code
/// \author Andreas Hoecker


#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TStopwatch.h"

#if not defined(__CINT__) || defined(__MAKECINT__)
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#endif

// two types of category methods are implemented
Bool_t UseOffsetMethod = kTRUE;

void TMVAClassificationCategoryApplication()
{
   // ---------------------------------------------------------------
   // default MVA methods to be trained + tested
   std::map<std::string,int> Use;
   //
   Use["LikelihoodCat"] = 1;
   Use["FisherCat"]     = 1;
   // ---------------------------------------------------------------

   std::cout << std::endl
             << "==> Start TMVAClassificationCategoryApplication" << std::endl;

   //  Create the Reader object

   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

   // Create a set of variables and spectators and declare them to the reader
   // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
   Float_t var1, var2, var3, var4, eta;
   reader->AddVariable( "var1", &var1 );
   reader->AddVariable( "var2", &var2 );
   reader->AddVariable( "var3", &var3 );
   reader->AddVariable( "var4", &var4 );

   reader->AddSpectator( "eta", &eta );

   // Book the MVA methods

   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = it->first + " method";
         TString weightfile = "dataset/weights/TMVAClassificationCategory_" + TString(it->first) + ".weights.xml";
         reader->BookMVA( methodName, weightfile );
      }
   }

   // Book output histograms
   UInt_t nbin = 100;
   std::map<std::string,TH1*> hist;
   hist["LikelihoodCat"] = new TH1F( "MVA_LikelihoodCat",   "MVA_LikelihoodCat", nbin, -1, 0.9999 );
   hist["FisherCat"]     = new TH1F( "MVA_FisherCat",       "MVA_FisherCat",     nbin, -4, 4 );

   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //
   TString fname = gSystem->GetDirName(__FILE__) + "/data/";
   // if directory data not found try using tutorials dir
   if (gSystem->AccessPathName( fname + "toy_sigbkg_categ_offset.root"  )) {
      fname = gROOT->GetTutorialDir() + "/tmva/data/";
   }
   if (UseOffsetMethod) fname += "toy_sigbkg_categ_offset.root";
   else                 fname += "toy_sigbkg_categ_varoff.root";
   std::cout << "--- TMVAClassificationApp    : Accessing " << fname << "!" << std::endl;
   TFile *input = TFile::Open(fname);
   if (!input) {
      std::cout << "ERROR: could not open data file: " << fname << std::endl;
      exit(1);
   }

   // Event loop

   // Prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = (TTree*)input->Get("TreeS");
   std::cout << "--- Use signal sample for evalution" << std::endl;
   theTree->SetBranchAddress( "var1", &var1 );
   theTree->SetBranchAddress( "var2", &var2 );
   theTree->SetBranchAddress( "var3", &var3 );
   theTree->SetBranchAddress( "var4", &var4 );

   theTree->SetBranchAddress( "eta",  &eta ); // spectator

   std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;
   TStopwatch sw;
   sw.Start();
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0) std::cout << "--- ... Processing event: " << ievt << std::endl;

      theTree->GetEntry(ievt);

      // Return the MVA outputs and fill into histograms

      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
         if (!it->second) continue;
         TString methodName = it->first + " method";
         hist[it->first]->Fill( reader->EvaluateMVA( methodName ) );
      }

   }
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();

   // Write histograms

   TFile *target  = new TFile( "TMVApp.root","RECREATE" );
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++)
      if (it->second) hist[it->first]->Write();

   target->Close();
   std::cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << std::endl;

   delete reader;
   std::cout << "==> TMVAClassificationApplication is done!" << std::endl << std::endl;
}

int main( int argc, char** argv )
{
   TMVAClassificationCategoryApplication();
   return 0;
}
