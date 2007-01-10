/*

  This program will add histograms and Trees from a list of root files and write them
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
   
  Authors: Rene Brun, Dirk Geppert, Sven A. Schmidt, sven.schmidt@cern.ch
         : rewritten from scratch by Rene Brun (30 November 2005)
            to support files with nested directories.
           Toby Burnett implemented the possibility to use indirect files.
 */

#include "RConfig.h"
#include <string>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TClass.h"

TList *FileList;
TFile *Target, *Source;
Bool_t noTrees;
Bool_t fastMethod;

int AddFile(TList* sourcelist, std::string entry, int newcomp) ;
void MergeRootfile( TDirectory *target, TList *sourcelist, Int_t isdir );

int main( int argc, char **argv ) {

   if ( argc < 4 || "-h" == string(argv[1]) || "--help" == string(argv[1]) ) {
      cout << "Usage: " << argv[0] << " [-f] [-T] targetfile source1 source2 [source3 ...]" << endl;
      cout << "This program will add histograms from a list of root files and write them" << endl;
      cout << "to a target root file. The target file is newly created and must not " << endl;
      cout << "exist, or if -f (\"force\") is given, must not be one of the source files." << endl;
      cout << "Supply at least two source files for this to make sense... ;-)" << endl;
      cout << "If the first argument is -T, Trees are not merged" <<endl;
      cout << "When -the -f option is specified, one can also specify the compression" <<endl;
      cout << "level of the target file. By default the compression level is 1, but" <<endl;
      cout << "if \"-f0\" is specified, the target file will not be compressed." <<endl;
      cout << "if \"-f6\" is specified, the compression level 6 will be used." <<endl;
      cout << "if Target and source files have different compression levels"<<endl;
      cout << " a slower method is used"<<endl;
      return 1;
   }
   FileList = new TList();

   Bool_t force = (!strcmp(argv[1],"-f") || !strcmp(argv[2],"-f"));
   noTrees = (!strcmp(argv[1],"-T") || !strcmp(argv[2],"-T"));
   Int_t newcomp = 1;
   char ft[4];
   for (int j=0;j<9;j++) {
      sprintf(ft,"-f%d",j);
      if (!strcmp(argv[1],ft) || !strcmp(argv[2],ft)) {
         force = kTRUE;
          newcomp = j;
         break;
      }
   }
  
   int ffirst = 2;
   if (force) ffirst++;
   if (noTrees) ffirst++;

   cout << "Target file: " << argv[ffirst-1] << endl;
   Target = TFile::Open( argv[ffirst-1], (force?"RECREATE":"CREATE") );
   if (!Target || Target->IsZombie()) {
      cerr << "Error opening target file (does " << argv[ffirst-1] << " exist?)." << endl;
      cerr << "Pass \"-f\" argument to force re-creation of output file." << endl;
      exit(1);
   }
   Target->SetCompressionLevel(newcomp);
  
   // by default hadd can merge Trees in a file that can go up to 100 Gbytes
   Long64_t maxsize = 100000000; //100GB
   maxsize *= 100;  //to bypass some compiler limitations with big constants
   TTree::SetMaxTreeSize(maxsize);
  
   fastMethod = kTRUE;
   for ( int i = ffirst; i < argc; i++ ) {
      if( AddFile(FileList, argv[i], newcomp) !=0 ) return 1;
   }
   if (!fastMethod) {
      cout <<"Sources and Target have different compression levels"<<endl;
      cout <<"Merging will be slower"<<endl;
   }

   MergeRootfile( Target, FileList,0 );

   //must delete Target to avoid a problem with dictionaries in~ TROOT
   delete Target;

   return 0;
}

int AddFile(TList* sourcelist, std::string entry, int newcomp) {
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

    TFile* Source = TFile::Open( entry.c_str());
    if( Source==0 ){
        return 1;
    }
    sourcelist->Add(Source);
    if (newcomp != Source->GetCompressionLevel())  fastMethod = kFALSE;
    return 0;
}


void MergeRootfile( TDirectory *target, TList *sourcelist, Int_t isdir ) {

   cout << "Target path: " << target->GetPath() << endl;
   TString path( (char*)strstr( target->GetPath(), ":" ) );
   path.Remove( 0, 2 );

   TDirectory *first_source = (TDirectory*)sourcelist->First();
   THashList allNames;
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
         if (allNames.FindObject(key->GetName())) continue;
         allNames.Add(new TObjString(key->GetName()));
            
         // read object from first source file
         current_sourcedir->cd();
         TObject *obj = key->ReadObj();

         if ( obj->IsA()->InheritsFrom( TH1::Class() ) ) {
            // descendant of TH1 -> merge it

            TH1 *h1 = (TH1*)obj;
            TList listH;

            // loop over all source files and add the content of the
            // correspondant histogram to the one pointed to by "h1"
            TFile *nextsource = (TFile*)sourcelist->After( first_source );
            while ( nextsource ) {
               // make sure we are at the correct directory level by cd'ing to path
               TDirectory *ndir = nextsource->GetDirectory(path);
               if (ndir) {
                  ndir->cd();
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(h1->GetName());
                  if (key2) {
                     TObject *hobj = key2->ReadObj();
                     hobj->ResetBit(kMustCleanup);
                     listH.Add(hobj);
                     h1->Merge(&listH);
                     listH.Delete();
                  }
               }
               nextsource = (TFile*)sourcelist->After( nextsource );
            }
         } else if ( obj->IsA()->InheritsFrom( "TTree" ) ) {
      
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
         } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
            // it's a subdirectory

            cout << "Found subdirectory " << obj->GetName() << endl;
            // create a new subdir of same name and title in the target file
            target->cd();
            TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

            // newdir is now the starting point of another round of merging
            // newdir still knows its depth within the target file via
            // GetPath(), so we can still figure out where we are in the recursion
            MergeRootfile( newdir, sourcelist,1);

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
            if(obj->IsA()->InheritsFrom( "TDirectory" )) {
               //printf("cas d'une directory\n");
            } else if(obj->IsA()->InheritsFrom( "TTree" )) {
               if (!noTrees) {
                  globChain->ls();
                  if (fastMethod) globChain->Merge(target->GetFile(),0,"keep fast");
                  else            globChain->Merge(target->GetFile(),0,"keep");
                  delete globChain;
               }
            } else {
               obj->Write( key->GetName() );
            }
         }
         oldkey = key;
      } // while ( ( TKey *key = (TKey*)nextkey() ) )
      first_source = (TDirectory*)sourcelist->After(first_source);
   }
   // save modifications to target file
   target->SaveSelf(kTRUE);
   if (!isdir) sourcelist->Remove(sourcelist->First());
}
