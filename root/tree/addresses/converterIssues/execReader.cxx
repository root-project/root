
// System include(s):
#include <cstdlib>
#include <iostream>
#include <memory>

// ROOT include(s):
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TError.h>

#define AODX_STANDALONE 1 

// Local include(s):
#include "EventFormat.h"

#include "EventFormatElement.cxx"
#include "EventFormat.cxx"

#include "selection_LinkDef.h"

int execReader() {

   // Name for the application:
   static const char* APP_NAME = "fileReader";

   // Try to open the test input file:
   std::unique_ptr< TFile > ifile( TFile::Open( "test_file.root", "READ" ) );
   if( ( ! ifile.get() ) || ifile->IsZombie() ) {
      Error( APP_NAME, "Couldn't open test file!" );
      Error( APP_NAME, "Make sure that you run fileWriter first." );
      return EXIT_FAILURE;
   }
   Info( APP_NAME, "Opened the input file" );

   // Access the TTree in the file:
   TTree* itree = dynamic_cast< TTree* >( ifile->Get( "MetaData" ) );
   if( ! itree ) {
      Error( APP_NAME, "Could not find the TTree in the input file!" );
      return EXIT_FAILURE;
   }
   Info( APP_NAME, "Input tree accessed" );

   // Access the EventFormat branch:
   edm::EventFormat* format = 0; TBranch* br = 0;
   if( itree->SetBranchAddress( "EventFormat", &format, &br ) < 0 ) {
      Error( APP_NAME, "Couldn't connect to the EventFormat branch!" );
      return EXIT_FAILURE;
   }

   // Print the contents of the smart object:
   if( br->GetEntry( 0 ) <= 0 ) {
      Error( APP_NAME,
             "Failed to read in the first entry for branch EventFormat" );
      return EXIT_FAILURE;
   }
   std::cout << *format << std::endl;

   // Return gracefully:
   return EXIT_SUCCESS;
}
