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
              fPrintLevel(0), fMaxOpenedFiles( R__GetSystemMaxOpenedFiles() ),
              fLocal(isLocal), fHistoOneGo(histoOneGo)
{
   // Create file merger object.

   fFileList = new TList;
   fFileList->SetOwner(kTRUE);

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
      Printf("Source file %d: %s",fFileList->GetEntries()+1,url);
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
      
      fFileList->Add(newfile);
      
      TObjString *urlObj = new TObjString(url);
      fMergeList->Add(urlObj);
      
      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TFileMerger::AddAdoptFile(TFile *source, Bool_t cpProgress)
{
   // Add the TFile to this file merger and give ownership of the TFile to this
   // object (unless kFALSE is returned).
   // 
   // Return kTRUE if the addition was successful.
   
   if (source == 0) {
      return kFALSE;
   }

   if (fPrintLevel > 0) {
      Printf("Source file %d: %s",fFileList->GetEntries()+1,source->GetName());
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
      
      fFileList->Add(newfile);
      
      if (!fMergeList) {
         fMergeList = new TList;
      }
      TObjString *urlObj = new TObjString(source->GetName());
      fMergeList->Add(urlObj);
      
      if (newfile != source) {
         delete source;
      }
      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, Bool_t force, Int_t compressionLevel)
{
   // Open merger output file.
   
   fExplicitCompLevel = kTRUE;

   SafeDelete(fOutputFile);
   
   fOutputFilename = outputfile;
   
   if (!(fOutputFile = TFile::Open(outputfile, (force?"RECREATE":"CREATE"), "", compressionLevel)) || fOutputFile->IsZombie()) {
      Error("OutputFile", "cannot open the MERGER output file %s", fOutputFilename.Data());
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile, Bool_t force /* = kTRUE */ )
{
   // Open merger output file.
   
   Bool_t res = OutputFile(outputfile,force,1); // 1 is the same as the default from the TFile constructor.
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
Bool_t TFileMerger::IncrementalMerge(Bool_t)
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
   
   fOutputFile->SetBit(kMustCleanup);
   
   TDirectory::TContext ctxt(0);
   
   Bool_t result = kTRUE;
   while (result && fFileList->GetEntries()>0) {
      result = MergeRecursive(fOutputFile, fFileList, kTRUE);
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
      fFileList->Clear();
      if (fExcessFiles->GetEntries() > 0) {
         OpenExcessFiles();
      }
   }
   if (!result) {
      Error("Merge", "error during merge of your ROOT files");
   } else {
      fOutputFile->Write("",TObject::kOverwrite);
   }
   Clear();
   return result;
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

   TDirectory::TContext ctxt(0);
   
   fOutputFile->SetBit(kMustCleanup);
   Bool_t result = kTRUE;
   while (result && fFileList->GetEntries()>0) {
      result = MergeRecursive(fOutputFile, fFileList);

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
      fFileList->Clear();
      if (fExcessFiles->GetEntries() > 0) {
         OpenExcessFiles();         
      }
   }
   if (!result) {
      Error("Merge", "error during merge of your ROOT files");
   } else {
      // But Close is required so the file is complete.
      fOutputFile->Close();
   }
   
   // Cleanup
   fOutputFile->ResetBit(kMustCleanup);
   SafeDelete(fOutputFile);
   return result;
}

//______________________________________________________________________________
Bool_t TFileMerger::MergeRecursive(TDirectory *target, TList *sourcelist, Bool_t incremental /* = kFALSE */)
{
   // Merge all objects in a directory
   // If incremental is true, accumulate the data into the already existing objects in target.

   Bool_t status = kTRUE;
   if (fPrintLevel > 0) {
      Printf("Target path: %s",target->GetPath());
   }

   // Get the dir name
   TString path(target->GetPath());
   path.Remove(0, path.Last(':') + 2);

   // Gain time, do not add the objects in the list in memory.
   Bool_t addDirStat = kTRUE;
   if (R__TH1_Class) {
      gROOT->ProcessLineFast("TH1::AddDirectoryStatus()");
      gROOT->ProcessLine("TH1::AddDirectory(kFALSE);");
   }
   
   TDirectory *first_source = 0;
   if (!incremental) {
      first_source = (TDirectory*)sourcelist->First();
   }
   
   Int_t nguess = sourcelist->GetSize()+1000;
   THashList allNames(nguess);
   allNames.SetOwner(kTRUE);
   ((THashList*)target->GetList())->Rehash(nguess);
   ((THashList*)target->GetListOfKeys())->Rehash(nguess);
   
   TFileMergeInfo info(target);

   if ((fFastMethod && !fCompressionChange)) {
      info.fOptions.Append(" fast");
   }

   TDirectory *current_sourcedir;
   if (incremental) {
      current_sourcedir = target;
   } else {
      current_sourcedir = first_source->GetDirectory(path);
   }
   while (first_source || current_sourcedir) {
      // When current_sourcedir != 0 and firstsource==0 we are going over the target
      // for an incremental merge.
      if (current_sourcedir && (first_source == 0 || current_sourcedir != target)) {

         // loop over all keys in this directory
         TIter nextkey( current_sourcedir->GetListOfKeys() );
         TKey *key;
         TString oldkeyname;
         
         while ( (key = (TKey*)nextkey())) {
            
            // Keep only the highest cycle number for each key.
            if (oldkeyname == key->GetName()) continue;
            // Read in but do not copy directly the processIds.
            if (strcmp(key->GetClassName(),"TProcessID") == 0) { key->ReadObj(); continue;}
            // We already seen this object [name] and thus we already processed
            // the whole list of files for this objects.
            if (allNames.FindObject(key->GetName())) continue;
            
            TClass *cl = TClass::GetClass(key->GetClassName());
            if (!cl || !cl->InheritsFrom(TObject::Class())) {
               Info("MergeRecursive", "cannot merge object type, name: %s title: %s",
                    key->GetName(), key->GetTitle());
               continue;
            }
            allNames.Add(new TObjString(key->GetName()));
            
            if (fNoTrees && cl->InheritsFrom(R__TTree_Class)) {
               continue;
            }
            
            // read object from first source file
            TObject *obj;
            if (incremental) {
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
               
               //DIFFERENCE: hadd used to issue an information message:
               //    cout << "Found subdirectory " << obj->GetName() << endl;
               // create a new subdir of same name and title in the target file
               target->cd();
               TDirectory *newdir;
               if (incremental) {
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
               status = MergeRecursive( newdir, sourcelist);
               if (!status) return status;
               
            } else if (obj->IsA()->GetMerge()) {
               
               TList inputs;
               Bool_t oneGo = fHistoOneGo && obj->IsA()->InheritsFrom(R__TH1_Class);
               
               // Loop over all source files and merge same-name object
               TFile *nextsource = first_source ? (TFile*)sourcelist->After( first_source ) : (TFile*)sourcelist->First();
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
               TFile *nextsource = first_source ? (TFile*)sourcelist->After( first_source ) : (TFile*)sourcelist->First();
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
               TFile *nextsource = first_source ? (TFile*)sourcelist->After( first_source ) : (TFile*)sourcelist->First();
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
               Warning("MergeRecursive", "cannot merge object type (n:'%s', t:'%s') - "
                       "Merge(TCollection *) not implemented",
                       obj->GetName(), obj->GetTitle());
               
               // Loop over all source files and write similar objects directly to the output file
               TFile *nextsource = first_source ? (TFile*)sourcelist->After( first_source ) : (TFile*)sourcelist->First();
               while (nextsource) {
                  // make sure we are at the correct directory level by cd'ing to path
                  TDirectory *ndir = nextsource->GetDirectory(path);
                  if (ndir) {
                     ndir->cd();
                     TKey *key2 = (TKey*)ndir->GetListOfKeys()->FindObject(key->GetName());
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
      first_source = first_source ? (TDirectory*)sourcelist->After(first_source) : (TFile*)sourcelist->First();
      if (first_source) {
         current_sourcedir = first_source->GetDirectory(path);
      } else {
         current_sourcedir = 0;
      }
   }
   // save modifications to target file
   target->SaveSelf(kTRUE);
   if (R__TH1_Class) {
      gROOT->ProcessLine(TString::Format("TH1::AddDirectory(%d);",addDirStat));
   }
   return status;
}

//______________________________________________________________________________
Bool_t TFileMerger::OpenExcessFiles()
{
   // Open up to fMaxOpenedFiles of the excess files.
   
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
