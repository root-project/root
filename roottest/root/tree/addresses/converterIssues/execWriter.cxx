
// System include(s):
#include <cstdlib>

// ROOT include(s):
#include <TFile.h>
#include <TTree.h>
#include <TError.h>
#include <TRandom.h>

#define AODX_STANDALONE 1 

// Local include(s):
#include "EventFormat_p1.h"

#include "EventFormatElement.cxx"
#include "EventFormat_p1.cxx"
#include "selection_p1_LinkDef.h"

int execWriter() {

   // Name for the application:
   static const char* APP_NAME = "fileWriter";

   // Open an output file:
   TFile* ofile = TFile::Open( "test_file.root", "RECREATE" );
   if( ( ! ofile ) || ofile->IsZombie() ) {
      Error( APP_NAME, "Couldn't open output file!" );
      return EXIT_FAILURE;
   }
   ofile->cd();
   Info( APP_NAME, "Opened the output file" );

   // Create a TTree in it:
   TTree* otree = new TTree( "MetaData", "TestTree" );
   otree->SetAutoSave( 100000000 );
   otree->SetDirectory( ofile );
   Info( APP_NAME, "Created a TTree called \"CollectionTree\"" );

   // Create the object that will be written out:
   EventFormat_p1* format = new EventFormat_p1();
   format->m_branchNames.push_back( "Test1" );
   format->m_classNames.push_back( "Bla1" );
   format->m_branchHashes.push_back( 1 );
   format->m_branchNames.push_back( "Test2" );
   format->m_classNames.push_back( "Bla2" );
   format->m_branchHashes.push_back( 2 );

   // Write it in a branch to the tree:
   if( ! otree->Branch( "EventFormat", &format ) ) {
      Error( APP_NAME, "Couldn't create EventFormat branch" );
      return EXIT_FAILURE;
   }

   // Fill the event:
   otree->Fill();

   // Write out the tree:
   otree->Write();
   otree->SetDirectory( 0 );
   delete otree;

   // Delete the output object:
   delete format;

   // Close the output file:
   ofile->Close();
   delete ofile;
   Info( APP_NAME, "Output file closed" );

   // Return gracefully:
   return EXIT_SUCCESS;
}
