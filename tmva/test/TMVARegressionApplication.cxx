/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVARegressionApplication                                          *
 *                                                                                *
 * This macro provides a simple example on how to use the trained regression MVAs *
 * within an analysis module                                                      *
 **********************************************************************************/

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
#include "TPluginManager.h"
#include "TStopwatch.h"

#include "TMVA/Reader.h"
#include "TMVA/Tools.h"
#include "TMVA/MethodCuts.h"

// read input data file with ascii format (otherwise ROOT) ?
Bool_t ReadDataFromAsciiIFormat = kFALSE;

int main( int argc, char** argv ) 
{
   //---------------------------------------------------------------
   // default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   Use["PDERS"]           = 1;
   Use["PDERSkNN"]        = 0; 
   Use["PDEFoam"]         = 1; 
   // ---
   Use["KNN"]             = 0;
   // ---
   Use["LD"]		         = 1;
   // ---
   Use["FDA_GA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 1;
   Use["FDA_GAMT"]        = 0;
   // ---
   Use["MLP"]             = 1; 
   // ---
   Use["BDT"]             = 0;
   Use["BDTG"]            = 0;
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVARegressionApplication" << std::endl;

   std::cout << "Running the following methods" << std::endl;
   if (argc>1) {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;
   }
   for (int i=1; i<argc; i++) {
      std::string regMethod(argv[i]);
      if (Use.find(regMethod) == Use.end()) {
         std::cout << "Method " << regMethod << " not known in TMVA under this name. Please try one of:" << std::endl;
         for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) std::cout << it->first << " ";
         std::cout << std::endl;
         return 1;
      }
      Use[regMethod] = kTRUE;
   }

   //
   // create the Reader object
   //
   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

   // create a set of variables and declare them to the reader
   // - the variable names must corresponds in name and type to 
   // those given in the weight file(s) that you use
   Float_t var1, var2;
   reader->AddVariable( "var1", &var1 );
   reader->AddVariable( "var2", &var2 );

   //Spectator variables declared in the training have to be added to the reader, too
   Float_t spec1,spec2;
   reader->AddSpectator( "spec1:=var1*2",  &spec1 );
   reader->AddSpectator( "spec2:=var1*3",  &spec2 );

   //
   // book the MVA methods
   //
   TString dir    = "weights/";
   TString prefix = "TMVARegression";

   // book method(s)
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = it->first + " method";
         TString weightfile = dir + prefix + "_" + TString(it->first) + ".weights.xml";
         reader->BookMVA( methodName, weightfile ); 
      }
   }
   
   // example how to use your own method as plugin
   if (Use["Plugin"]) {
      // the weight file contains a line 
      // Method         : MethodName::InstanceName

      // if MethodName is not a known TMVA method, it is assumed to be
      // a user implemented method which has to be loaded via the
      // plugin mechanism
      
      // for user implemented methods the line in the weight file can be
      // Method         : PluginName::InstanceName
      // where PluginName can be anything

      // before usage the plugin has to be defined, which can happen
      // either through the following line in .rootrc:
      // # plugin handler          plugin       class            library        constructor format
      // Plugin.TMVA@@MethodBase:  PluginName   MethodClassName  UserPackage    "MethodName(DataSet&,TString)"
      //  
      // or by telling the global plugin manager directly
      gPluginMgr->AddHandler("TMVA@@MethodBase", "PluginName", "MethodClassName", "UserPackage", "MethodName(DataSet&,TString)");
      // the class is then looked for in libUserPackage.so

      // now the method can be booked like any other
      reader->BookMVA( "User method", dir + prefix + "_User.weights.txt" );
   }

   // book output histograms
   std::vector<TH1*> hists;
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      TH1* h = new TH1F( it->first.c_str(), TString(it->first) + " method", 100, -100, 600 );
      std::cout << "booking hist: " << it->first.c_str() << std::endl;
      if (it->second) hists.push_back( h );
   }
   
   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   //   
   TFile *input(0);
   TString fname = "./tmva_reg_example.root";
   if (!gSystem->AccessPathName( fname )) {
      // first we try to find tmva_example.root in the local directory
      std::cout << "--- TMVARegressionApp        : Accessing data file \"" << fname << "\"" << std::endl;
      input = TFile::Open( fname );
   } 
   else { 
      // finally we try accessing the file via the web from
      // http://root.cern.ch/files/tmva_example.root
      std::cout << "ERROR: cannot access data file: " << fname << std::endl;
      exit(1);
   }
   if (!input) {
      std::cout << "ERROR: could not open data file: " << fname << std::endl;
      exit(1);
   }

   //
   // prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = (TTree*)input->Get("TreeR");
   std::cout << "--- Select signal sample" << std::endl;
   theTree->SetBranchAddress( "var1", &var1 );
   theTree->SetBranchAddress( "var2", &var2 );

   std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;
   TStopwatch sw;
   sw.Start();
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0) {
         std::cout << "--- ... Processing event: " << ievt << std::endl;
      }

      theTree->GetEntry(ievt);

      // 
      // retrieve the MVA target values (regression outputs) and fill into histograms
      // NOTE: EvaluateRegression(..) returns a vector for multi-target regression
      // 
      for (std::vector<TH1*>::iterator it = hists.begin(); it != hists.end(); it++) {
         (*it)->Fill( reader->EvaluateRegression( (*it)->GetTitle() )[0] );            
      }
   }
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();

   //
   // write histograms
   //
   TFile *target  = new TFile( "TMVARegApp.root","RECREATE" );
   for (std::vector<TH1*>::iterator it = hists.begin(); it != hists.end(); it++) (*it)->Write();
   target->Close();

   std::cout << "--- Created root file: \"" << target->GetName() 
             << "\" containing the MVA output histograms" << std::endl;
  
   delete reader;
    
   std::cout << "==> TMVARegressionApplication is done!" << std::endl << std::endl;
}
