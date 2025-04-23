
/**********************************************************************

        Simple example demonstrating some sort of problem with
        TFileMerger.

 **********************************************************************/

// STL include(s):
#include <iostream>

// ROOT include(s):
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <TFileMerger.h>
#include "TROOT.h"


using namespace std;

// Forward declare helper function:
bool writeSimpleFile( const char* fileName );

/**
 * Main function demonstrating the problem.
 */
void runfilemergererror() {

   //
   // Create two simple ROOT files holding a TTree:
   //
   if (1) {
      if( ! writeSimpleFile( "TestFile1.root" ) ) {
         cerr << "There was a problem writing TestFile1.root" << endl;
         return;
      } else {
         cout << "Written file: TestFile1.root" << endl;
      }
      if( ! writeSimpleFile( "TestFile2.root" ) ) {
         cerr << "There was a problem writing TestFile2.root" << endl;
         return;
      } else {
         cout << "Written file: TestFile2.root" << endl;
      }
   }

   //
   // Merge these two files using TFileMerger:
   //
   TFileMerger merger( kFALSE );
   merger.AddFile( "TestFile1.root" );
   merger.AddFile( "TestFile2.root" );
   merger.OutputFile( "TestMerged.root" );
   merger.Merge(kFALSE);
   
   //
   // Test that the merged file "looks okay":
   //
   TFile* rfile = TFile::Open( "TestMerged.root", "READ" );
   if( ! rfile ) {
      cerr << "Couldn't open file: TestMerged.root" << endl;
      return;
   } else {
      cout << "Opened file: TestMerged.root" << endl;
   }
   TTree* rtree = ( TTree* ) rfile->Get( "TestTree" );
   if( ! rtree ) {
      cerr << "Couldn't access TestTree!" << endl;
      return;
   } else {
      cout << "Accessed TestTree" << endl;
   }
   //rtree->Print();
   delete rfile;

   //
   // Open the file in update mode. ROOT will give a warning...
   //
   TFile* ufile = TFile::Open( "TestMerged.root", "UPDATE" );
   if( ! ufile ) {
      cerr << "Couldn't open file: TestMerged.root in update mode" << endl;
      return;
   } else {
      cout << "Opened file: TestMerged.root in update mode" << endl;
   }
   TTree* utree = ( TTree* ) ufile->Get( "TestTree" );
   if( ! utree ) {
      cerr << "Couldn't access TestTree!" << endl;
      return;
   } else {
      cout << "Accessed TestTree" << endl;
   }
   if( ufile ) delete ufile;

   return;

}

/**
 * Helper function creating a ROOT file with name "fileName",
 * holding very simple TTree called "TestTree".
 */
bool writeSimpleFile( const char* fileName ) {

   //
   // Open the output file:
   //
   TFile* ofile = TFile::Open( fileName, "RECREATE" );
   if( ! ofile ) {
      cerr << "Couldn't open file \"" << fileName << "\" for writing!"
           << endl;
      return false;
   }

   //
   // Create a simple TTree in the output file:
   //
   const Int_t branchStyle = 1;
   const Int_t autoSave = 10000000;
   TTree* tree = new TTree( "TestTree", "Very simple TTree" );
   tree->SetAutoSave( autoSave );
   TTree::SetBranchStyle( branchStyle );
   tree->SetDirectory( ofile );

   //
   // Declare two branches in the TTree:
   //
   Int_t intOutput;
   Double_t doubleOutput;
   tree->Branch( "iout", &intOutput, "iout/I" );
   tree->Branch( "dout", &doubleOutput, "dout/D" );

   //
   // Fill the TTree with 1000 entries:
   //
   for( Int_t i = 0; i < 1000; ++i ) {
      intOutput = i;
      doubleOutput = gRandom->Rndm();
      tree->Fill();
   }

   //
   // Close the file:
   //
   tree->AutoSave();
   delete tree;
   ofile->Write();
   ofile->Close();
   delete ofile;

   return true;

}
