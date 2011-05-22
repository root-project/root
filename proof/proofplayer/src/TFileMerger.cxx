// @(#)root/proofplayer:$Id$
// Author: Andreas Peters + Fons Rademakers + Rene Brun  26/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileMerger                                                          //
//                                                                      //
// This class provides file copy and merging services.                  //
//                                                                      //
// It can be used to copy files (not only ROOT files), using TFile or   //
// any of its remote file access plugins. It is therefore usefull in    //
// a Grid environment where the files might be accessable via Castor,   //
// rfio, dcap, etc.                                                     //
// The merging interface allows files containing histograms and trees   //
// to be merged, like the standalone hadd program.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFileMerger.h"
#include "TUrl.h"
#include "TFile.h"
#include "TUUID.h"
#include "TSystem.h"
#include "TH1.h"
#include "THStack.h"
#include "TChain.h"
#include "TKey.h"
#include "THashList.h"
#include "TObjString.h"
#include "TClass.h"
#include "TMethodCall.h"
#include "Riostream.h"


ClassImp(TFileMerger)

//______________________________________________________________________________
TFileMerger::TFileMerger(Bool_t isLocal, Bool_t histoOneGo)
            : fOutputFile(0), fFastMethod(kTRUE), fNoTrees(kFALSE), fExplicitCompLevel(kFALSE), fCompressionChange(kFALSE),
              fLocal(isLocal), fHistoOneGo(histoOneGo)
{
   // Create file merger object.

   fFileList = new TList;
   fFileList->SetOwner(kTRUE);

   fMergeList = new TList;
   fMergeList->SetOwner(kTRUE);
}

//______________________________________________________________________________
TFileMerger::~TFileMerger()
{
   // Cleanup.

   SafeDelete(fFileList);
   SafeDelete(fMergeList);
   SafeDelete(fOutputFile);
}

//______________________________________________________________________________
void TFileMerger::Reset()
{
   // Reset merger file list.

   fFileList->Clear();
   fMergeList->Clear();
}

//______________________________________________________________________________
Bool_t TFileMerger::AddFile(const char *url, Bool_t cpProgress)
{
   // Add file to file merger.

   TFile *newfile = 0;
   TString localcopy;

   if (fLocal) {
      TUUID uuid;
      localcopy.Form("file:%s/ROOTMERGE-%s.root", gSystem->TempDirectory(), uuid.AsString());
      if (!TFile::Cp(url, localcopy, cpProgress)) {
         Error("AddFile", "cannot get a local copy of file %s", url);
         return kFALSE;
      }
      newfile = TFile::Open(localcopy, "READ");
   } else {
      newfile = TFile::Open(url, "READ");
   }

   if (!newfile) {
      if (fLocal)
         Error("AddFile", "cannot open local copy %s of URL %s",
                          localcopy.Data(), url);
      else
         Error("AddFile", "cannot open file %s", url);
      return kFALSE;
   } else {
      if (fOutputFile && fOutputFile->GetCompressionLevel() != newfile->GetCompressionLevel()) fCompressionChange = kTRUE;

      fFileList->Add(newfile);

      if (!fMergeList)
         fMergeList = new TList;
      TObjString *urlObj = new TObjString(url);
      fMergeList->Add(urlObj);

      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, Int_t compressionLevel)
{
   // Open merger output file.
   
   fExplicitCompLevel = kTRUE;

   SafeDelete(fOutputFile);
   
   fOutputFilename = outputfile;
   
   if (!(fOutputFile = TFile::Open(outputfile, "RECREATE", "", compressionLevel))) {
      Error("OutputFile", "cannot open the MERGER output file %s", fOutputFilename.Data());
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile)
{
   // Open merger output file.
   
   Bool_t res = OutputFile(outputfile,1); // 1 is the same as the default from the TFile constructor.
   fExplicitCompLevel = kFALSE;
   return res;
}

//______________________________________________________________________________
void TFileMerger::PrintFiles(Option_t *options)
{
   // Print list of files being merged.

   fFileList->Print(options);
}

//______________________________________________________________________________
Bool_t TFileMerger::Merge(Bool_t)
{
   // Merge the files. If no output file was specified it will write into
   // the file "FileMerger.root" in the working directory. Returns true
   // on success, false in case of error.

   if (!fOutputFile) {
      TString outf(fOutputFilename);
      if (outf.IsNull()) {
         outf.Form("file:%s/FileMerger.root", gSystem->TempDirectory());
         Info("Merge", "will merge the results to the file %s\n"
                       "since you didn't specify a merge filename",
                       TUrl(outf).GetFile());
      }
      if (!OutputFile(outf.Data())) {
         return kFALSE;
      }
   }

   Bool_t result = MergeRecursive(fOutputFile, fFileList);
   if (!result) {
      Error("Merge", "error during merge of your ROOT files");
   } else {
      // But Close is required so the file is complete.
      fOutputFile->Close();
   }

   // Cleanup
   SafeDelete(fOutputFile);

   // Remove local copies if there are any
   TIter next(fFileList);
   TFile *file;
   while ((file = (TFile*) next())) {
      // close the files
      file->Close();
      // remove the temporary files
      if(fLocal) {
         TString p(file->GetPath());
         p = p(0, p.Index(':',0));
         gSystem->Unlink(p);
      }
   }
   return result;
}

//______________________________________________________________________________
Bool_t TFileMerger::MergeRecursive(TDirectory *target, TList *sourcelist)
{
   // Merge all objects in a directory
   // NB. This function is a copy of the hadd function MergeROOTFile

   Bool_t status = kTRUE;

   // Get the dir name
   TString path(target->GetPath());
   path.Remove(0, path.Last(':') + 2);

   //gain time, do not add the objects in the list in memory
   Bool_t addDirStat = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);

   TDirectory *first_source = (TDirectory*)sourcelist->First();

   Int_t nguess = sourcelist->GetSize()+1000;
   THashList allNames(nguess);
   ((THashList*)target->GetList())->Rehash(nguess);
   ((THashList*)target->GetListOfKeys())->Rehash(nguess);

   while (first_source) {
      TDirectory *current_sourcedir = first_source->GetDirectory(path);
      if (!current_sourcedir) {
         first_source = (TDirectory*)sourcelist->After(first_source);
         continue;
      }

      // loop over all keys in this directory
      TChain *globChain = 0;
      TIter nextkey( current_sourcedir->GetListOfKeys() );
      TKey *key, *oldkey=0;

      while ( (key = (TKey*)nextkey())) {
         if (current_sourcedir == target) break;
         //keep only the highest cycle number for each key
         if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;
         if (!strcmp(key->GetClassName(),"TProcessID")) {key->ReadObj(); continue;}
         if (allNames.FindObject(key->GetName())) continue;
         TClass *cl = TClass::GetClass(key->GetClassName());
         if (!cl || !cl->InheritsFrom(TObject::Class())) {
            Info("MergeRecursive", "cannot merge object type, name: %s title: %s",
                                   key->GetName(), key->GetTitle());
            continue;
         }
         allNames.Add(new TObjString(key->GetName()));

         // read object from first source file
         TObject *obj = key->ReadObj();
         if (!obj) {
            Info("MergeRecursive", "could not read object for key {%s, %s}",
                                   key->GetName(), key->GetTitle());
            continue;
         }

         if (obj->IsA()->InheritsFrom(TH1::Class())) {
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
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(key->GetName());
                  if (key2) {
                     TObject *hobj = key2->ReadObj();
                     if (!hobj) {
                        Info("MergeRecursive", "could not read object for key {%s, %s}; skipping file %s",
                                               key->GetName(), key->GetTitle(), nextsource->GetName());
                        nextsource = (TFile*)sourcelist->After(nextsource);
                        continue;
                     }
                     hobj->ResetBit(kMustCleanup);
                     listH.Add(hobj);
                     // Run the merging now, if required
                     if (!fHistoOneGo) {
                        h1->Merge(&listH);
                        listH.Delete();
                     }
                  }
               }
               nextsource = (TFile*)sourcelist->After( nextsource );
            }
            // Merge the list, if still to be done
            if (fHistoOneGo) {
               h1->Merge(&listH);
               listH.Delete();
            }
         } else if ( obj->IsA()->InheritsFrom( TTree::Class() ) ) {

            // loop over all source files create a chain of Trees "globChain"
            if (!fNoTrees) {
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

            //cout << "Found subdirectory " << obj->GetName() << endl;
            // create a new subdir of same name and title in the target file
            target->cd();
            TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

            // newdir is now the starting point of another round of merging
            // newdir still knows its depth within the target file via
            // GetPath(), so we can still figure out where we are in the recursion
            status = MergeRecursive( newdir, sourcelist);
            if (!status) return status;

         } else if (obj->InheritsFrom(TObject::Class()) &&
                    obj->IsA()->GetMethodWithPrototype("Merge", "TCollection*") ) {
            // Object implements Merge(TCollection*)

            TList listH;
            TString listHargs;
            listHargs.Form("((TCollection*)0x%lx)", (ULong_t)&listH);

            // Loop over all source files and merge same-name object
            TFile *nextsource = (TFile*)sourcelist->After( first_source );
            while (nextsource) {
               // make sure we are at the correct directory level by cd'ing to path
               TDirectory *ndir = nextsource->GetDirectory(path);
               if (ndir) {
                  ndir->cd();
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(key->GetName());
                  if (key2) {
                     TObject *hobj = key2->ReadObj();
                     if (!hobj) {
                        Info("MergeRecursive", "could not read object for key {%s, %s}; skipping file %s",
                                               key->GetName(), key->GetTitle(), nextsource->GetName());
                        nextsource = (TFile*)sourcelist->After(nextsource);
                        continue;
                     }
                     // Set ownership for collections
                     if (hobj->InheritsFrom(TCollection::Class())) {
                        ((TCollection*)hobj)->SetOwner();
                     }
                     hobj->ResetBit(kMustCleanup);
                     listH.Add(hobj);
                     Int_t error = 0;
                     obj->Execute("Merge", listHargs.Data(), &error);
                     if (error) {
                        Error("MergeRecursive", "calling Merge() on '%s' with the corresponding object in '%s'",
                                                 obj->GetName(), nextsource->GetName());
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
                     if (!hstack2) {
                        Info("MergeRecursive", "could not read THStack for key {%s, %s}; skipping file %s",
                                               key->GetName(), key->GetTitle(), nextsource->GetName());
                        nextsource = (TFile*)sourcelist->After(nextsource);
                        continue;
                     }
                     l->Add(hstack2->GetHists()->Clone());
                     delete hstack2;
                  }
               }

               nextsource = (TFile*)sourcelist->After( nextsource );
            }
            hstack1->GetHists()->Merge(l);
            l->Delete();
         } else {
            // Object is of no type that we can merge
            Warning("MergeRecursive", "cannot merge object type (n:'%s', t:'%s') - "
                                      "Merge(TCollection *) not implemented",
                                      obj->GetName(), obj->GetTitle());

            // Loop over all source files and write similar objects directly to the output file
            TFile *nextsource = (TFile*)sourcelist->After( first_source );
            while (nextsource) {
               // make sure we are at the correct directory level by cd'ing to path
               TDirectory *ndir = nextsource->GetDirectory(path);
               if (ndir) {
                  ndir->cd();
                  TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(key->GetName());
                  if (key2) {
                     TObject *nobj = key2->ReadObj();
                     if (!nobj) {
                        Info("MergeRecursive", "could not read object for key {%s, %s}; skipping file %s",
                                               key->GetName(), key->GetTitle(), nextsource->GetName());
                        nextsource = (TFile*)sourcelist->After(nextsource);
                        continue;
                     }
                     nobj->ResetBit(kMustCleanup);
                     if (target->WriteTObject(nobj, key2->GetName(), "SingleKey") <= 0) {
                        Warning("MergeRecursive", "problems copying object (n:'%s', t:'%s') to output file ",
                                                  obj->GetName(), obj->GetTitle());
                        status = kFALSE;
                     }
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
            if (!fNoTrees) {
               if (globChain) {
                  globChain->ls("noaddr");
                  if (fFastMethod && !fCompressionChange) globChain->Merge(target->GetFile(),0,"keep fast");
                  else                                    globChain->Merge(target->GetFile(),0,"keep");
                  delete globChain;
               }
            }
         } else if (obj->IsA()->InheritsFrom( TCollection::Class() )) {
            if ( obj->Write( key->GetName(), TObject::kSingleKey ) <= 0 ) {
               status = kFALSE;
            }
            ((TCollection*)obj)->SetOwner();
         } else {
            if ( obj->Write( key->GetName() ) <= 0) {
               status = kFALSE;
            }
         }
         if (obj->IsA()->InheritsFrom(TCollection::Class())) ((TCollection*)obj)->Delete();
         oldkey = key;
         delete obj;
      } // while ( ( TKey *key = (TKey*)nextkey() ) )
      first_source = (TDirectory*)sourcelist->After(first_source);
   }
   // save modifications to target file
   target->SaveSelf(kTRUE);
   TH1::AddDirectory(addDirStat);
   return status;
}
