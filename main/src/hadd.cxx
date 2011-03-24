/*

  This program will add histograms (see note) and Trees from a list of root files and write them
  to a target root file. The target file is newly created and must not be
  identical to one of the source files.

  Syntax:

       hadd targetfile source1 source2 ...
    or
       hadd -f targetfile source1 source2 ...
         (targetfile is overwritten if it exists)

  When -the -f option is specified, one can also specify the compression
  level of the target file. By default the compression level is 1, but
  if "-f0" is specified, the target file will not be compressed.
  if "-f6" is specified, the compression level 6 will be used.

  For example assume 3 files f1, f2, f3 containing histograms hn and Trees Tn
    f1 with h1 h2 h3 T1
    f2 with h1 h4 T1 T2
    f3 with h5
   the result of
     hadd -f x.root f1.root f2.root f3.root
   will be a file x.root with h1 h2 h3 h4 h5 T1 T2
   where h1 will be the sum of the 2 histograms in f1 and f2
         T1 will be the merge of the Trees in f1 and f2

   The files may contain sub-directories.

  if the source files contains histograms and Trees, one can skip
  the Trees with
       hadd -T targetfile source1 source2 ...

  Wildcarding and indirect files are also supported
    hadd result.root  myfil*.root
   will merge all files in myfil*.root
    hadd result.root file1.root @list.txt file2. root myfil*.root
    will merge file1. root, file2. root, all files in myfil*.root
    and all files in the indirect text file list.txt ("@" as the first
    character of the file indicates an indirect file. An indirect file
    is a text file containing a list of other files, including other
    indirect files, one line per file).

  If the sources and and target compression levels are identical (default),
  the program uses the TChain::Merge function with option "fast", ie
  the merge will be done without  unzipping or unstreaming the baskets
  (i.e. direct copy of the raw byte on disk). The "fast" mode is typically
  5 times faster than the mode unzipping and unstreaming the baskets.

  NOTE1: By default histograms are added. However hadd does not support the case where
         histograms have their bit TH1::kIsAverage set.

  NOTE2: hadd returns a status code: 0 if OK, -1 otherwise

  Authors: Rene Brun, Dirk Geppert, Sven A. Schmidt, sven.schmidt@cern.ch
         : rewritten from scratch by Rene Brun (30 November 2005)
            to support files with nested directories.
           Toby Burnett implemented the possibility to use indirect files.
 */

#include "RConfig.h"
#include <string>
#include "TChain.h"
#include "TFile.h"
#include "THashList.h"
#include "TH1.h"
#include "THStack.h"
#include "TKey.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TClass.h"
#include "TSystem.h"
#include <stdlib.h>

TList *FileList;
TFile *Target, *Source;
Bool_t noTrees;
Bool_t fastMethod;
Bool_t reoptimize;

int AddFile(TList* sourcelist, std::string entry, int newcomp) ;
int MergeRootfile( TDirectory *target, TList *sourcelist);

//___________________________________________________________________________
int main( int argc, char **argv )
{

   if ( argc < 3 || "-h" == string(argv[1]) || "--help" == string(argv[1]) ) {
      cout << "Usage: " << argv[0] << " [-f[0-9]] [-k] [-T] [-O] targetfile source1 [source2 source3 ...]" << endl;
      cout << "This program will add histograms from a list of root files and write them" << endl;
      cout << "to a target root file. The target file is newly created and must not " << endl;
      cout << "exist, or if -f (\"force\") is given, must not be one of the source files." << endl;
      cout << "Supply at least two source files for this to make sense... ;-)" << endl;
      cout << "If the option -k is used, hadd will not exit on corrupt or non-existant input files but skip the offending files instead." << endl;
      cout << "If the option -T is used, Trees are not merged" <<endl;
      cout << "If the option -O is used, when merging TTree, the basket size is re-optimized" <<endl;
      cout << "When -the -f option is specified, one can also specify the compression" <<endl;
      cout << "level of the target file. By default the compression level is 1, but" <<endl;
      cout << "if \"-f0\" is specified, the target file will not be compressed." <<endl;
      cout << "if \"-f6\" is specified, the compression level 6 will be used." <<endl;
      cout << "if Target and source files have different compression levels"<<endl;
      cout << " a slower method is used"<<endl;
      return 1;
   }
   FileList = new TList();

   Bool_t force = kFALSE;
   Bool_t skip_errors = kFALSE;
   reoptimize = kFALSE;
   noTrees = kFALSE;

   int ffirst = 2;
   Int_t newcomp = 1;
   for( int a = 1; a < argc; ++a ) {
      if ( strcmp(argv[a],"-T") == 0 ) {
         noTrees = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-f") == 0 ) {
         force = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-k") == 0 ) {
         skip_errors = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-O") == 0 ) {
         reoptimize = kTRUE;
         ++ffirst;
      } else if ( argv[a][0] == '-' ) {
         char ft[4];
         for( int j=0; j<=9; ++j ) {
            snprintf(ft,4,"-f%d",j);
            if (!strcmp(argv[a],ft)) {
               force = kTRUE;
               newcomp = j;
               ++ffirst;
               break;
            }
         }
      }
   }

   gSystem->Load("libTreePlayer");

   cout << "Target file: " << argv[ffirst-1] << endl;

   Target = TFile::Open( argv[ffirst-1], (force?"RECREATE":"CREATE") );
   if (!Target || Target->IsZombie()) {
      cerr << "Error opening target file (does " << argv[ffirst-1] << " exist?)." << endl;
      cerr << "Pass \"-f\" argument to force re-creation of output file." << endl;
      exit(1);
   }
   Target->SetCompressionLevel(newcomp);

   // by default hadd can merge Trees in a file that can go up to 100 Gbytes
   // No need to set this, as 100Gb is now the TTree default
   // Long64_t maxsize = 100000000; //100GB
   // maxsize *= 1000;  //to bypass some compiler limitations with big constants
   // TTree::SetMaxTreeSize(maxsize);

   fastMethod = kTRUE;
   for ( int i = ffirst; i < argc; i++ ) {
      if( AddFile(FileList, argv[i], newcomp) !=0 ) {
         if ( skip_errors ) {
            cerr << "Skipping file with error: " << argv[i] << endl;
         } else {
	    cerr << "Exiting due to error in " << argv[i] << endl;
	    return 1;
         }
      }
   }
   if (!fastMethod && !reoptimize) {
      // Don't warn if the user any request re-optimization.
      cout <<"Sources and Target have different compression levels"<<endl;
      cout <<"Merging will be slower"<<endl;
   }

   int status = MergeRootfile( Target, FileList);

   //must delete Target to avoid a problem with dictionaries in~ TROOT
   delete Target;

   return status;
}

//___________________________________________________________________________
int AddFile(TList* sourcelist, std::string entry, int newcomp)
{
   // add a new file to the list of files
   static int count(0);
   if( entry.empty() ) return 0;
   size_t j =entry.find_first_not_of(' ');
   if( j==std::string::npos ) return 0;
   entry = entry.substr(j);
   if( entry.substr(0,1)=="@"){
      std::ifstream indirect_file(entry.substr(1).c_str() );
      if( ! indirect_file.is_open() ) {
         std::cerr<< "Could not open indirect file " << entry.substr(1) << std::endl;
         return 1;
      }
      while( indirect_file ){
         std::string line;
         std::getline(indirect_file, line);
         if( AddFile(sourcelist, line, newcomp)!=0 )return 1;;
      }
      return 0;
   }
   cout << "Source file " << (++count) << ": " << entry << endl;

   TFile* source = TFile::Open( entry.c_str());
   if( source==0 ){
      cerr << "Could not open file " << entry << endl;
      return 1;
   } else if ( source->IsZombie() ) {
      cerr << "Could not properly read file " << entry << endl;
      return 1;
   }
   sourcelist->Add(source);
   if (newcomp != source->GetCompressionLevel()) fastMethod = kFALSE;
   return 0;
}


//___________________________________________________________________________
int MergeRootfile( TDirectory *target, TList *sourcelist)
{
   // Merge all objects in a directory
   int status = 0;
   cout << "Target path: " << target->GetPath() << endl;
   TString path( (char*)strstr( target->GetPath(), ":" ) );
   path.Remove( 0, 2 );

   TDirectory *first_source = (TDirectory*)sourcelist->First();
   Int_t nguess = sourcelist->GetSize()+1000;
   THashList allNames(nguess);
   ((THashList*)target->GetList())->Rehash(nguess);
   ((THashList*)target->GetListOfKeys())->Rehash(nguess);
   TList listH;
   TString listHargs;
   listHargs.Form("((TCollection*)0x%lx)", (ULong_t)&listH);
   while(first_source) {
      TDirectory *current_sourcedir = first_source->GetDirectory(path);
      if (!current_sourcedir) {
         first_source = (TDirectory*)sourcelist->After(first_source);
         continue;
      }

      // loop over all keys in this directory
      TChain *globChain = 0;
      TIter nextkey( current_sourcedir->GetListOfKeys() );
      TKey *key, *oldkey=0;
      //gain time, do not add the objects in the list in memory
      TH1::AddDirectory(kFALSE);

      while ( (key = (TKey*)nextkey())) {
         if (current_sourcedir == target) break;
         //keep only the highest cycle number for each key
         if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;
         if (!strcmp(key->GetClassName(),"TProcessID")) {key->ReadObj(); continue;}
         if (allNames.FindObject(key->GetName())) continue;
         TClass *cl = TClass::GetClass(key->GetClassName());
         if (!cl || !cl->InheritsFrom(TObject::Class())) {
            cout << "Cannot merge object type, name: "
                 << key->GetName() << " title: " << key->GetTitle() << endl;
            continue;
         }
         allNames.Add(new TObjString(key->GetName()));
         // read object from first source file
         //current_sourcedir->cd();
         TObject *obj = key->ReadObj();
         //printf("keyname=%s, obj=%x\n",key->GetName(),obj);

         if ( obj->IsA()->InheritsFrom( TTree::Class() ) ) {

            // loop over all source files create a chain of Trees "globChain"
            if (!noTrees) {
               TString obj_name;
               if (path.Length()) {
                  obj_name = path + "/" + obj->GetName();
               } else {
                  obj_name = obj->GetName();
               }
               globChain = new TChain(obj_name);
               globChain->Add(first_source->GetName());
               TFile *nextsource = (TFile*)sourcelist->After( first_source );
               while ( nextsource ) {
                  //do not add to the list a file that does not contain this Tree
                  TFile *curf = TFile::Open(nextsource->GetName());
                  if (curf) {
                     Bool_t mustAdd = kFALSE;
                     if (curf->FindKey(obj_name)) {
                        mustAdd = kTRUE;
                     } else {
                        //we could be more clever here. No need to import the object
                        //we are missing a function in TDirectory
                        TObject *aobj = curf->Get(obj_name);
                        if (aobj) { mustAdd = kTRUE; delete aobj;}
                     }
                     if (mustAdd) {
                        globChain->Add(nextsource->GetName());
                     }
                  }
                  delete curf;
                  nextsource = (TFile*)sourcelist->After( nextsource );
               }
            }
         } else if ( obj->IsA()->InheritsFrom( TDirectory::Class() ) ) {
            // it's a subdirectory

            cout << "Found subdirectory " << obj->GetName() << endl;
            // create a new subdir of same name and title in the target file
            target->cd();
            TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

            // newdir is now the starting point of another round of merging
            // newdir still knows its depth within the target file via
            // GetPath(), so we can still figure out where we are in the recursion
            status = MergeRootfile( newdir, sourcelist);
            if (status) return status;

         } else if ( obj->InheritsFrom(TObject::Class())
              && obj->IsA()->GetMethodWithPrototype("Merge", "TCollection*") ) {
            // object implements Merge(TCollection*)

            // loop over all source files and merge same-name object
            TFile *nextsource = (TFile*)sourcelist->After( first_source );
            while ( nextsource ) {
               // make sure we are at the correct directory level by cd'ing to path
               TDirectory *ndir = nextsource->GetDirectory(path);
               if (ndir) {
                  ndir->cd();
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(key->GetName());
                  if (key2) {
                     TObject *hobj = key2->ReadObj();
                     hobj->ResetBit(kMustCleanup);
                     listH.Add(hobj);
                     Int_t error = 0;
                     obj->Execute("Merge", listHargs.Data(), &error);
                     if (error) {
                        cerr << "Error calling Merge() on " << obj->GetName()
                             << " with the corresponding object in " << nextsource->GetName() << endl;
                     }
                     listH.Delete();
                  }
               }
               nextsource = (TFile*)sourcelist->After( nextsource );
            }
         } else if ( obj->IsA()->InheritsFrom( THStack::Class() ) ) {
            THStack *hstack1 = (THStack*) obj;
            TList* l = new TList();

            // loop over all source files and merge the histos of the
            // corresponding THStacks with the one pointed to by "hstack1"
            TFile *nextsource = (TFile*)sourcelist->After( first_source );
            while ( nextsource ) {
               // make sure we are at the correct directory level by cd'ing to path
               TDirectory *ndir = nextsource->GetDirectory(path);
               if (ndir) {
                  ndir->cd();
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(hstack1->GetName());
                  if (key2) {
                    THStack *hstack2 = (THStack*) key2->ReadObj();
                    l->Add(hstack2->GetHists()->Clone());
                    delete hstack2;
                  }
               }

               nextsource = (TFile*)sourcelist->After( nextsource );
            }
            hstack1->GetHists()->Merge(l);
            l->Delete();
         } else {
            // object is of no type that we can merge
            cout << "Cannot merge object type, name: "
                 << obj->GetName() << " title: " << obj->GetTitle() << endl;

            // loop over all source files and write similar objects directly to the output file
            TFile *nextsource = (TFile*)sourcelist->After( first_source );
            while ( nextsource ) {
               // make sure we are at the correct directory level by cd'ing to path
               TDirectory *ndir = nextsource->GetDirectory(path);
               if (ndir) {
                  ndir->cd();
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(key->GetName());
                  if (key2) {
                     TObject *nobj = key2->ReadObj();
                     nobj->ResetBit(kMustCleanup);
                     int nbytes1 = target->WriteTObject(nobj, key2->GetName(), "SingleKey" );
                     if (nbytes1 <= 0) status = -1;
                     delete nobj;
                  }
               }
               nextsource = (TFile*)sourcelist->After( nextsource );
            }
         }

         // now write the merged histogram (which is "in" obj) to the target file
         // note that this will just store obj in the current directory level,
         // which is not persistent until the complete directory itself is stored
         // by "target->Write()" below
         target->cd();

         //!!if the object is a tree, it is stored in globChain...
         if(obj->IsA()->InheritsFrom( TDirectory::Class() )) {
            //printf("cas d'une directory\n");
         } else if(obj->IsA()->InheritsFrom( TTree::Class() )) {
            if (!noTrees) {
               globChain->ls("noaddr");
               if (fastMethod && !reoptimize) globChain->Merge(target->GetFile(),0,"keep fast");
               else                           globChain->Merge(target->GetFile(),0,"keep");
               delete globChain;
            }
         } else {
            int nbytes2 = obj->Write( key->GetName(), TObject::kSingleKey );
            if (nbytes2 <= 0) status = -1;
         }
         oldkey = key;
         delete obj;
      } // while ( ( TKey *key = (TKey*)nextkey() ) )
      first_source = (TDirectory*)sourcelist->After(first_source);
   }
   // save modifications to target file
   target->SaveSelf(kTRUE);
   return status;
}
