/*

  This program will add histograms from a list of root files and write them
  to a target root file. The target file is newly created and must not be
  identical to one of the source files.

  Syntax:

  hadd targetfile source1 source2 ...

  Author: Sven A. Schmidt, sven.schmidt@cern.ch
  Date:   13.2.2001

  This code is based on the hadd.C example by Rene Brun and Dirk Geppert,
  which had a problem with directories more than one level deep.
  (see macro hadd_old.C for this previous implementation).

  I have tested this macro on rootfiles with one and two dimensional
  histograms, and two levels of subdirectories. Feel free to send comments
  or bug reports to me.

 */

#include <string>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "Riostream.h"

TList *FileList;
TFile *Target;

void MergeRootfile( TDirectory *target, TList *sourcelist );

int main( int argc, char **argv ) {

  if ( argc < 4 || "-h" == string(argv[1]) || "--help" == string(argv[1]) ) {
    cout << "Usage: " << argv[0] << " targetfile source1 source2 [source3 ...]" << endl;
    cout << "This program will add histograms from a list of root files and write them" << endl;
    cout << "to a target root file. The target file is newly created and must not be" << endl;
    cout << "identical to one of the source files." << endl;
    cout << "Supply at least two source files for this to make sense... ;-)" << endl;
    return 1;
  }
   FileList = new TList();

  cout << "Target file: " << argv[1] << endl;
  Target = TFile::Open( argv[1], "RECREATE" );

  for ( int i = 2; i < argc; i++ ) {
    cout << "Source file " << i-1 << ": " << argv[i] << endl;
    FileList->Add( TFile::Open( argv[i] ) );
  }

  MergeRootfile( Target, FileList );

  return 0;
}

void MergeRootfile( TDirectory *target, TList *sourcelist ) {

  //  cout << "Target path: " << target->GetPath() << endl;
  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );

  TFile *first_source = (TFile*)sourcelist->First();
  first_source->cd( path );
  TDirectory *current_sourcedir = gDirectory;

  // loop over all keys in this directory
  TChain *globChain = 0;
  TIter nextkey( current_sourcedir->GetListOfKeys() );
  TKey *key, *oldkey=0;
  while ( (key = (TKey*)nextkey())) {

    //keep only the highest cycle number for each key
    if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;
     
    // read object from first source file
    first_source->cd( path );
    TObject *obj = key->ReadObj();

    if ( obj->IsA()->InheritsFrom( "TH1" ) ) {
      // descendant of TH1 -> merge it

      //      cout << "Merging histogram " << obj->GetName() << endl;
      TH1 *h1 = (TH1*)obj;

      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      while ( nextsource ) {
        
        // make sure we are at the correct directory level by cd'ing to path
        nextsource->cd( path );
        TH1 *h2 = (TH1*)gDirectory->Get( h1->GetName() );
        if ( h2 ) {
          h1->Add( h2 );
          delete h2; // don't know if this is necessary, i.e. if 
                     // h2 is created by the call to gDirectory above.
        }

        nextsource = (TFile*)sourcelist->After( nextsource );
      }
    }
    else if ( obj->IsA()->InheritsFrom( "TTree" ) ) {
      
      // loop over all source files create a chain of Trees "globChain"
      TString obj_name;
      if (path.Length()) {
         obj_name = path + "/" + obj->GetName();
      } else {
         obj_name = obj->GetName();
      }

      globChain = new TChain(obj_name);
      globChain->Add(first_source->GetName());
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      //      const char* file_name = nextsource->GetName();
      // cout << "file name  " << file_name << endl;
      while ( nextsource ) {
     	  
         globChain->Add(nextsource->GetName());
         nextsource = (TFile*)sourcelist->After( nextsource );
      }

    } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory

      cout << "Found subdirectory " << obj->GetName() << endl;

      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

      // newdir is now the starting point of another round of merging
      // newdir still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      MergeRootfile( newdir, sourcelist );

    } else {

      // object is of no type that we know or can handle
      cout << "Unknown object type, name: " 
           << obj->GetName() << " title: " << obj->GetTitle() << endl;
    }

    // now write the merged histogram (which is "in" obj) to the target file
    // note that this will just store obj in the current directory level,
    // which is not persistent until the complete directory itself is stored
    // by "target->Write()" below
    if ( obj ) {
       target->cd();
       
       //!!if the object is a tree, it is stored in globChain...
       if(obj->IsA()->InheritsFrom( "TTree" ))
          globChain->Merge(target->GetFile(),0,"keep");
       else
          obj->Write( key->GetName() );
    }
    oldkey = key;

  } // while ( ( TKey *key = (TKey*)nextkey() ) )

  // save modifications to target file
  target->SaveSelf(kTRUE);

}
