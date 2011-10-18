// @(#)root/io:$Id$
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
#include "TKey.h"
#include "THashList.h"
#include "TObjString.h"
#include "TClass.h"
#include "TMethodCall.h"
#include "Riostream.h"
#include "TFileMergeInfo.h"
#include "TClassRef.h"
#include "TROOT.h"
#include "TMemFile.h"

#ifdef WIN32
// For _getmaxstdio
#include <stdio.h> 
#else
// For getrlimit
#include <sys/time.h>
#include <sys/resource.h>
#endif

ClassImp(TFileMerger)

TClassRef R__TH1_Class("TH1");
TClassRef R__TTree_Class("TTree");

static const Int_t kCpProgress = BIT(14);
static const Int_t kCintFileNumber = 100;
//______________________________________________________________________________
static Int_t R__GetSystemMaxOpenedFiles()
{
   // Return the maximum number of allowed opened files minus some wiggle room
   // for CINT or at least of the standard library (stdio).
   int maxfiles;
#ifdef WIN32
   maxfiles = _getmaxstdio();
#else
   rlimit filelimit;
   if (getrlimit(RLIMIT_NOFILE,&filelimit)==0) {
      maxfiles = filelimit.rlim_cur;
   } else {
      // We could not get the value from getrlimit, let's return a reasonable default.
      maxfiles = 512;
   }
#endif
   if (maxfiles > kCintFileNumber) {
      return maxfiles - kCintFileNumber;
   } else if (maxfiles > 5) {
      return maxfiles - 5;
   } else {
      return maxfiles;
   }
}

//______________________________________________________________________________
TFileMerger::TFileMerger(Bool_t isLocal, Bool_t histoOneGo)
            : fOutputFile(0), fFastMethod(kTRUE), fNoTrees(kFALSE), fExplicitCompLevel(kFALSE), fCompressionChange(kFALSE),
              fPrintLevel(0), fMsgPrefix("TFileMerger"), fMaxOpenedFiles( R__GetSystemMaxOpenedFiles() ),
              fLocal(isLocal), fHistoOneGo(histoOneGo)
{
   // Create file merger object.

   fFileList = new TList;

   fMergeList = new TList;
   fMergeList->SetOwner(kTRUE);
   
   fExcessFiles = new TList;
   fExcessFiles->SetOwner(kTRUE);
   
   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TFileMerger::~TFileMerger()
{
   // Cleanup.

   gROOT->GetListOfCleanups()->Remove(this);
   SafeDelete(fFileList);
   SafeDelete(fMergeList);
   SafeDelete(fOutputFile);
   SafeDelete(fExcessFiles);
}

//______________________________________________________________________________
void TFileMerger::Reset()
{
   // Reset merger file list.

   fFileList->Clear();
   fMergeList->Clear();
   fExcessFiles->Clear();
}

//______________________________________________________________________________
Bool_t TFileMerger::AddFile(const char *url, Bool_t cpProgress)
{
   // Add file to file merger.
   
   if (fPrintLevel > 0) {
      Printf("%s Source file %d: %s",fMsgPrefix.Data(),fFileList->GetEntries()+fExcessFiles->GetEntries()+1,url);
   }
   
   TFile *newfile = 0;
   TString localcopy;
   
   if (fFileList->GetEntries() >= (fMaxOpenedFiles-1)) {

      TObjString *urlObj = new TObjString(url);
      fMergeList->Add(urlObj);

      urlObj = new TObjString(url);
      urlObj->SetBit(kCpProgress);
      fExcessFiles->Add(urlObj);
      
      return kTRUE;
   }
   
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
      
      newfile->SetBit(kCanDelete);
      fFileList->Add(newfile);
      
      TObjString *urlObj = new TObjString(url);
      fMergeList->Add(urlObj);
      
      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TFileMerger::AddFile(TFile *source, Bool_t cpProgress)
{
   // Add the TFile to this file merger and *do not* give ownership of the TFile to this
   // object.
   // 
   // Return kTRUE if the addition was successful.
   
   return AddFile(source,kFALSE,cpProgress);
}

//______________________________________________________________________________
Bool_t TFileMerger::AddAdoptFile(TFile *source, Bool_t cpProgress)
{
   // Add the TFile to this file merger and give ownership of the TFile to this
   // object (unless kFALSE is returned).
   // 
   // Return kTRUE if the addition was successful.

   return AddFile(source,kTRUE,cpProgress);
}

//______________________________________________________________________________
Bool_t TFileMerger::AddFile(TFile *source, Bool_t own, Bool_t cpProgress)
{
   // Add the TFile to this file merger and give ownership of the TFile to this
   // object (unless kFALSE is returned).
   // 
   // Return kTRUE if the addition was successful.
   
   if (source == 0) {
      return kFALSE;
   }

   if (fPrintLevel > 0) {
      Printf("%s Source file %d: %s",fMsgPrefix.Data(),fFileList->GetEntries()+1,source->GetName());
   }
   
   TFile *newfile = 0;
   TString localcopy;
   
   if (fLocal && !source->InheritsFrom(TMemFile::Class())) {
      TUUID uuid;
      localcopy.Form("file:%s/ROOTMERGE-%s.root", gSystem->TempDirectory(), uuid.AsString());
      if (!source->Cp(localcopy, cpProgress)) {
         Error("AddFile", "cannot get a local copy of file %s", source->GetName());
         return kFALSE;
      }
      newfile = TFile::Open(localcopy, "READ");
   } else {
      newfile = source;
   }
   
   if (!newfile) {
      if (fLocal)
         Error("AddFile", "cannot open local copy %s of URL %s",
               localcopy.Data(), source->GetName());
      else
         Error("AddFile", "cannot open file %s", source->GetName());
      return kFALSE;
   } else {
      if (fOutputFile && fOutputFile->GetCompressionLevel() != newfile->GetCompressionLevel()) fCompressionChange = kTRUE;
      
      if (own || newfile != source) {
         newfile->SetBit(kCanDelete);
      } else {
         newfile->ResetBit(kCanDelete);
      }
      fFileList->Add(newfile);
      
      if (!fMergeList) {
         fMergeList = new TList;
      }
      TObjString *urlObj = new TObjString(source->GetName());
      fMergeList->Add(urlObj);
      
      if (newfile != source && own) {
         delete source;
      }
      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, Bool_t force, Int_t compressionLevel)
{
   // Open merger output file.
   
   return OutputFile(outputfile,(force?"RECREATE":"CREATE"),compressionLevel);
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, Bool_t force)
{
   // Open merger output file.
   
   Bool_t res = OutputFile(outputfile,(force?"RECREATE":"CREATE"),1); // 1 is the same as the default from the TFile constructor.
   fExplicitCompLevel = kFALSE;
   return res;
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, const char *mode, Int_t compressionLevel)
{
   // Open merger output file.  'mode' is passed to the TFile constructor as the option, it should
   // be one of 'NEW','CREATE','RECREATE','UPDATE'
   // 'UPDATE' is usually used in conjunction with IncrementalMerge.
   
   fExplicitCompLevel = kTRUE;
   
   TFile *oldfile = fOutputFile;
   fOutputFile = 0; // This avoids the complaint from RecursiveRemove about the file being deleted which is here spurrious. (see RecursiveRemove).
   SafeDelete(oldfile);
   
   fOutputFilename = outputfile;
   
   if (!(fOutputFile = TFile::Open(outputfile, mode, "", compressionLevel)) || fOutputFile->IsZombie()) {
      Error("OutputFile", "cannot open the MERGER output file %s", fOutputFilename.Data());
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, const char *mode /* = "RECREATE" */)
{
   // Open merger output file.  'mode' is passed to the TFile constructor as the option, it should
   // be one of 'NEW','CREATE','RECREATE','UPDATE'
   // 'UPDATE' is usually used in conjunction with IncrementalMerge.

   Bool_t res = OutputFile(outputfile,mode,1); // 1 is the same as the default from the TFile constructor.
   fExplicitCompLevel = kFALSE;
   return res;
}

//______________________________________________________________________________
void TFileMerger::PrintFiles(Option_t *options)
{
   // Print list of files being merged.

   fFileList->Print(options);
   fExcessFiles->Print(options);
}

//______________________________________________________________________________
Bool_t TFileMerger::Merge(Bool_t)
{
   // Merge the files. If no output file was specified it will write into
   // the file "FileMerger.root" in the working directory. Returns true
   // on success, false in case of error.
   
   return PartialMerge(kAll | kRegular);
}

//______________________________________________________________________________
Bool_t TFileMerger::MergeRecursive(TDirectory *target, TList *sourcelist, Int_t type /* = kRegular | kAll */)
{
   // Merge all objects in a directory
   // The type is defined by the bit values in EPartialMergeType:
   //   kRegular      : normal merge, overwritting the output file (default)
   //   kIncremental  : merge the input file with the (existing) content of the output file (if already exising)
   //   kAll          : merge all type of objects (default)
   //   kResetable    : merge only the objects with a MergeAfterReset member function.
   //   kNonResetable : merge only the objects without a MergeAfterReset member function.

   Bool_t status = kTRUE;
   if (fPrintLevel > 0) {
      Printf("%s Target path: %s",fMsgPrefix.Data(),target->GetPath());
   }

   // Get the dir name
   TString path(target->GetPath());
   path.Remove(0, path.Last(':') + 2);

   Int_t nguess = sourcelist->GetSize()+1000;
   THashList allNames(nguess);
   allNames.SetOwner(kTRUE);
   ((THashList*)target->GetList())->Rehash(nguess);
   ((THashList*)target->GetListOfKeys())->Rehash(nguess);
   
   TFileMergeInfo info(target);

   if ((fFastMethod && !fCompressionChange)) {
      info.fOptions.Append(" fast");
   }

   TFile      *current_file;
   TDirectory *current_sourcedir;
   if (type & kIncremental) {
      current_file      = 0;
      current_sourcedir = target;
   } else {
      current_file      = (TFile*)sourcelist->First();
      current_sourcedir = current_file->GetDirectory(path);
   }
   while (current_file || current_sourcedir) {
      // When current_sourcedir != 0 and current_file == 0 we are going over the target
      // for an incremental merge.
      if (current_sourcedir && (current_file == 0 || current_sourcedir != target)) {

         // loop over all keys in this directory
         TIter nextkey( current_sourcedir->GetListOfKeys() );
         TKey *key;
         TString oldkeyname;
         
         while ( (key = (TKey*)nextkey())) {
            
            // Keep only the highest cycle number for each key.  They are stored in the (hash) list
            // consecutively and in decreasing order of cycles, so we can continue until the name
            // changes.
            if (oldkeyname == key->GetName()) continue;
            // Read in but do not copy directly the processIds.
            if (strcmp(key->GetClassName(),"TProcessID") == 0) { key->ReadObj(); continue;}
            // If we have already seen this object [name], we already processed
            // the whole list of files for this objects and we can just skip it
            // and any related cycles.
            if (allNames.FindObject(key->GetName())) {
               oldkeyname = key->GetName();
               continue;
            }
            
            TClass *cl = TClass::GetClass(key->GetClassName());
            if (!cl || !cl->InheritsFrom(TObject::Class())) {
               Info("MergeRecursive", "cannot merge object type, name: %s title: %s",
                    key->GetName(), key->GetTitle());
               continue;
            }
            allNames.Add(new TObjString(key->GetName()));
            
            if (fNoTrees && cl->InheritsFrom(R__TTree_Class)) {
               // Skip the TTree objects and any related cycles.
               oldkeyname = key->GetName();
               continue;
            }
            
            if (!(type&kResetable && type&kNonResetable)) {
               // If neither or both are requested at the same time, we merger both types.
               if (!(type&kResetable)) {
                  if (cl->GetResetAfterMerge()) {
                     // Skip the object with a reset after merge routine (TTree and other incrementally mergeable objects)
                     oldkeyname = key->GetName();
                     continue;                  
                  }
               }
               if (!(type&kNonResetable)) {
                  if (!cl->GetResetAfterMerge()) {
                     // Skip the object without a reset after merge routine (Histograms and other non incrementally mergeable objects)
                     oldkeyname = key->GetName();
                     continue;                  
                  }
               }
            }
            // read object from first source file
            TObject *obj;
            if (type & kIncremental) {
               obj = current_sourcedir->GetList()->FindObject(key->GetName());
               if (!obj) {
                  obj = key->ReadObj();
               }
            } else {
               obj = key->ReadObj();
            }
            if (!obj) {
               Info("MergeRecursive", "could not read object for key {%s, %s}",
                    key->GetName(), key->GetTitle());
               continue;
            }
            
            if ( obj->IsA()->InheritsFrom( TDirectory::Class() ) ) {
               // it's a subdirectory
               
               target->cd();
               TDirectory *newdir;
               if (type & kIncremental) {
                  newdir = target->GetDirectory(obj->GetName());
                  if (!newdir) {
                     newdir = target->mkdir( obj->GetName(), obj->GetTitle() );
                  }
               } else {
                  newdir = target->mkdir( obj->GetName(), obj->GetTitle() );
               }
               
               // newdir is now the starting point of another round of merging
               // newdir still knows its depth within the target file via
               // GetPath(), so we can still figure out where we are in the recursion
               status = MergeRecursive(newdir, sourcelist);
               if (!status) return status;
               
            } else if (obj->IsA()->GetMerge()) {
               
               TList inputs;
               Bool_t oneGo = fHistoOneGo && obj->IsA()->InheritsFrom(R__TH1_Class);
               
               // Loop over all source files and merge same-name object
               TFile *nextsource = current_file ? (TFile*)sourcelist->After( current_file ) : (TFile*)sourcelist->First();
               if (nextsource == 0) {
                  // There is only one file in the list
                  ROOT::MergeFunc_t func = obj->IsA()->GetMerge();
                  func(obj, &inputs, &info);
                  info.fIsFirst = kFALSE;
               } else {
                  do {
                     // make sure we are at the correct directory level by cd'ing to path
                     TDirectory *ndir = nextsource->GetDirectory(path);
                     if (ndir) {
                        ndir->cd();
                        TKey *key2 = (TKey*)ndir->GetListOfKeys()->FindObject(key->GetName());
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
                           inputs.Add(hobj);
                           if (!oneGo) {
                              ROOT::MergeFunc_t func = obj->IsA()->GetMerge();
                              Long64_t result = func(obj, &inputs, &info);
                              info.fIsFirst = kFALSE;
                              if (result < 0) {
                                 Error("MergeRecursive", "calling Merge() on '%s' with the corresponding object in '%s'",
                                       obj->GetName(), nextsource->GetName());
                              }
                              inputs.Delete();
                           }
                        }
                     }
                     nextsource = (TFile*)sourcelist->After( nextsource );
                  } while (nextsource);
                  // Merge the list, if still to be done
                  if (oneGo || info.fIsFirst) {
                     ROOT::MergeFunc_t func = obj->IsA()->GetMerge();
                     func(obj, &inputs, &info);
                     info.fIsFirst = kFALSE;
                     inputs.Delete();
                  }
               }
            } else if (obj->InheritsFrom(TObject::Class()) &&
                       obj->IsA()->GetMethodWithPrototype("Merge", "TCollection*,TFileMergeInfo*") ) {
               // Object implements Merge(TCollection*,TFileMergeInfo*) and has a reflex dictionary ... 
               
               TList listH;
               TString listHargs;
               listHargs.Form("(TCollection*)0x%lx,(TFileMergeInfo*)0x%lx", (ULong_t)&listH,(ULong_t)&info);
               
               // Loop over all source files and merge same-name object
               TFile *nextsource = current_file ? (TFile*)sourcelist->After( current_file ) : (TFile*)sourcelist->First();
               if (nextsource == 0) {
                  // There is only one file in the list
                  Int_t error = 0;
                  obj->Execute("Merge", listHargs.Data(), &error);
                  info.fIsFirst = kFALSE;
                  if (error) {
                     Error("MergeRecursive", "calling Merge() on '%s' with the corresponding object in '%s'",
                           obj->GetName(), key->GetName());
                  }
               } else {
                  while (nextsource) {
                     // make sure we are at the correct directory level by cd'ing to path
                     TDirectory *ndir = nextsource->GetDirectory(path);
                     if (ndir) {
                        ndir->cd();
                        TKey *key2 = (TKey*)ndir->GetListOfKeys()->FindObject(key->GetName());
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
                           info.fIsFirst = kFALSE;
                           if (error) {
                              Error("MergeRecursive", "calling Merge() on '%s' with the corresponding object in '%s'",
                                    obj->GetName(), nextsource->GetName());
                           }
                           listH.Delete();
                        }
                     }
                     nextsource = (TFile*)sourcelist->After( nextsource );
                  }
                  // Merge the list, if still to be done
                  if (info.fIsFirst) {
                     Int_t error = 0;
                     obj->Execute("Merge", listHargs.Data(), &error);
                     info.fIsFirst = kFALSE;
                     listH.Delete();
                  }
               }
            } else if (obj->InheritsFrom(TObject::Class()) &&
                       obj->IsA()->GetMethodWithPrototype("Merge", "TCollection*") ) {
               // Object implements Merge(TCollection*) and has a reflex dictionary ...
               
               TList listH;
               TString listHargs;
               listHargs.Form("((TCollection*)0x%lx)", (ULong_t)&listH);
               
               // Loop over all source files and merge same-name object
               TFile *nextsource = current_file ? (TFile*)sourcelist->After( current_file ) : (TFile*)sourcelist->First();
               if (nextsource == 0) {
                  // There is only one file in the list
                  Int_t error = 0;
                  obj->Execute("Merge", listHargs.Data(), &error);
                  if (error) {
                     Error("MergeRecursive", "calling Merge() on '%s' with the corresponding object in '%s'",
                           obj->GetName(), key->GetName());
                  }
               } else {
                  while (nextsource) {
                     // make sure we are at the correct directory level by cd'ing to path
                     TDirectory *ndir = nextsource->GetDirectory(path);
                     if (ndir) {
                        ndir->cd();
                        TKey *key2 = (TKey*)ndir->GetListOfKeys()->FindObject(key->GetName());
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
                           info.fIsFirst = kFALSE;
                           if (error) {
                              Error("MergeRecursive", "calling Merge() on '%s' with the corresponding object in '%s'",
                                    obj->GetName(), nextsource->GetName());
                           }
                           listH.Delete();
                        }
                     }
                     nextsource = (TFile*)sourcelist->After( nextsource );
                  }
                  // Merge the list, if still to be done
                  if (info.fIsFirst) {
                     Int_t error = 0;
                     obj->Execute("Merge", listHargs.Data(), &error);
                     info.fIsFirst = kFALSE;
                     listH.Delete();
                  }
               }
            } else {
               // Object is of no type that we can merge
               Bool_t warned = kFALSE;

               // Loop over all source files and write similar objects directly to the output file
               TFile *nextsource = current_file ? (TFile*)sourcelist->After( current_file ) : (TFile*)sourcelist->First();
               while (nextsource) {
                  // make sure we are at the correct directory level by cd'ing to path
                  TDirectory *ndir = nextsource->GetDirectory(path);
                  if (ndir) {
                     ndir->cd();
                     TKey *key2 = (TKey*)ndir->GetListOfKeys()->FindObject(key->GetName());
                     if (key2) {
                        if (warned) {
                           Warning("MergeRecursive", "cannot merge object type (n:'%s', t:'%s') - "
                                   "Merge(TCollection *) not implemented",
                                   obj->GetName(), obj->GetTitle());
                           warned = kTRUE;
                        }
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
            // by "target->SaveSelf()" below
            target->cd();
            
            oldkeyname = key->GetName();
            //!!if the object is a tree, it is stored in globChain...
            if(obj->IsA()->InheritsFrom( TDirectory::Class() )) {
               //printf("cas d'une directory\n");
            } else if (obj->IsA()->InheritsFrom( TCollection::Class() )) {
               if ( obj->Write( oldkeyname, TObject::kSingleKey | TObject::kOverwrite ) <= 0 ) {
                  status = kFALSE;
               }
               ((TCollection*)obj)->SetOwner();
            } else {
               if ( obj->Write( oldkeyname, TObject::kOverwrite ) <= 0) {
                  status = kFALSE;
               }
            }
            if (obj->IsA()->InheritsFrom(TCollection::Class())) ((TCollection*)obj)->Delete();
            delete obj;
            info.Reset();
         } // while ( ( TKey *key = (TKey*)nextkey() ) )
      }
      current_file = current_file ? (TFile*)sourcelist->After(current_file) : (TFile*)sourcelist->First();
      if (current_file) {
         current_sourcedir = current_file->GetDirectory(path);
      } else {
         current_sourcedir = 0;
      }
   }
   // save modifications to the target directory.
   if (! type&kIncremental) {
      // In case of incremental build, we will call Write on the top directory/file, so we do not need
      // to call SaveSelf explicilty.
      target->SaveSelf(kTRUE);
   }
   return status;
}

//______________________________________________________________________________
Bool_t TFileMerger::PartialMerge(Int_t in_type)
{
   // Merge the files. If no output file was specified it will write into
   // the file "FileMerger.root" in the working directory. Returns true
   // on success, false in case of error.
   // The type is defined by the bit values in EPartialMergeType:
   //   kRegular      : normal merge, overwritting the output file 
   //   kIncremental  : merge the input file with the content of the output file (if already exising) (default)
   //   kAll          : merge all type of objects (default)
   //   kResetable    : merge only the objects with a MergeAfterReset member function.
   //   kNonResetable : merge only the objects without a MergeAfterReset member function.
   //
   // If the type is set to kIncremental the output file is done deleted at the end of
   // this operation.  If the type is not set to kIncremental, the output file is closed.

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
   
   fOutputFile->SetBit(kMustCleanup);

   TDirectory::TContext ctxt(0);
   
   Bool_t result = kTRUE;
   Int_t type = in_type;
   while (result && fFileList->GetEntries()>0) {
      result = MergeRecursive(fOutputFile, fFileList, type);
      
      // Remove local copies if there are any
      TIter next(fFileList);
      TFile *file;
      while ((file = (TFile*) next())) {
         // close the files
         if (file->TestBit(kCanDelete)) file->Close();
         // remove the temporary files
         if(fLocal) {
            TString p(file->GetPath());
            p = p(0, p.Index(':',0));
            gSystem->Unlink(p);
         }
      }
      fFileList->Clear();
      if (fExcessFiles->GetEntries() > 0) {
         // We merge the first set of files in the output,
         // we now need to open the next set and make
         // sure we accumulate into the output, so we 
         // switch to incremental merging (if not already set)
         type = type | kIncremental;
         OpenExcessFiles();         
      }
   }
   if (!result) {
      Error("Merge", "error during merge of your ROOT files");
   } else {
      // Close or write is required so the file is complete.
      if (in_type & kIncremental) {
         fOutputFile->Write("",TObject::kOverwrite);
      } else {
         fOutputFile->Close();
      }
   }
   
   // Cleanup
   if (in_type & kIncremental) {
      Clear();
   } else {
      fOutputFile->ResetBit(kMustCleanup);
      SafeDelete(fOutputFile);
   }
   return result;
}

//______________________________________________________________________________
Bool_t TFileMerger::OpenExcessFiles()
{
   // Open up to fMaxOpenedFiles of the excess files.
   
   if (fPrintLevel > 0) {
      Printf("%s Opening the next %d files",fMsgPrefix.Data(),TMath::Min(fExcessFiles->GetEntries(),(fMaxOpenedFiles-1)));
   }   
   Int_t nfiles = 0;
   TIter next(fExcessFiles);
   TObjString *url = 0;
   TString localcopy;
   while( nfiles < (fMaxOpenedFiles-1) && ( url = (TObjString*)next() ) ) {
      TFile *newfile = 0;
      if (fLocal) {
         TUUID uuid;
         localcopy.Form("file:%s/ROOTMERGE-%s.root", gSystem->TempDirectory(), uuid.AsString());
         if (!TFile::Cp(url->GetName(), localcopy, url->TestBit(kCpProgress))) {
            Error("OpenExcessFiles", "cannot get a local copy of file %s", url->GetName());
            return kFALSE;
         }
         newfile = TFile::Open(localcopy, "READ");
      } else {
         newfile = TFile::Open(url->GetName(), "READ");
      }
      
      if (!newfile) {
         if (fLocal)
            Error("OpenExcessFiles", "cannot open local copy %s of URL %s",
                  localcopy.Data(), url->GetName());
         else
            Error("OpenExcessFiles", "cannot open file %s", url->GetName());
         return kFALSE;
      } else {
         if (fOutputFile && fOutputFile->GetCompressionLevel() != newfile->GetCompressionLevel()) fCompressionChange = kTRUE;
         
         newfile->SetBit(kCanDelete);
         fFileList->Add(newfile);
         ++nfiles;
         fExcessFiles->Remove(url);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
void TFileMerger::RecursiveRemove(TObject *obj)
{
   // Intercept the case where the output TFile is deleted!
   
   if (obj == fOutputFile) {
      Fatal("RecursiveRemove","Output file of the TFile Merger (targetting %s) has been deleted (likely due to a TTree larger than 100Gb)", fOutputFilename.Data());
   }
   
}

//______________________________________________________________________________
void TFileMerger::SetMaxOpenedFiles(Int_t newmax)
{
   // Set a limit to the number file that TFileMerger will opened at one time.
   // If the request is higher than the system limit, we reset it to the system limit.
   // If the request is less than two, we reset it to 2 (one for the output file and one for the input file).

   Int_t sysmax = R__GetSystemMaxOpenedFiles();
   if (newmax < sysmax) {
      fMaxOpenedFiles = newmax;
   } else {
      fMaxOpenedFiles = sysmax;
   }
   if (fMaxOpenedFiles < 2) {
      fMaxOpenedFiles = 2;
   }
}

//______________________________________________________________________________
void TFileMerger::SetMsgPrefix(const char *prefix)
{
   // Set the prefix to be used when printing informational message.

   fMsgPrefix = prefix;
}