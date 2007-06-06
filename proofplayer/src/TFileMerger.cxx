// @(#)root/proofplayer:$Name:  $:$Id: TFileMerger.cxx,v 1.14 2007/03/19 10:46:10 rdm Exp $
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
#include "TChain.h"
#include "TKey.h"
#include "THashList.h"
#include "TObjString.h"
#include "TClass.h"
#include "Riostream.h"

ClassImp(TFileMerger)

//______________________________________________________________________________
TFileMerger::TFileMerger() : fOutputFile(0),fFastMethod(kTRUE), fNoTrees(kFALSE)
{
   // Create file merger object.

   fFileList = new TList;
   fFileList->SetOwner(kTRUE);
}

//______________________________________________________________________________
TFileMerger::~TFileMerger()
{
   // Cleanup.

   if (fFileList)
      delete fFileList;

   if (fOutputFile)
      delete fOutputFile;
}

//______________________________________________________________________________
void TFileMerger::PrintProgress(Long64_t bytesread, Long64_t size)
{
   // Print file copy progress.

   fprintf(stderr, "[TFile::Cp] Total %.02f MB\t|", (Double_t)size/1048576);

   for (int l = 0; l < 20; l++) {
      if (size > 0) {
         if (l < 20*bytesread/size)
            fprintf(stderr, "=");
         else if (l == 20*bytesread/size)
            fprintf(stderr, ">");
         else if (l > 20*bytesread/size)
            fprintf(stderr, ".");
      } else
         fprintf(stderr, "=");
   }
   // Allow to update the GUI while uploading files
   gSystem->ProcessEvents();
   fWatch.Stop();
   Double_t lCopy_time = fWatch.RealTime();
   fprintf(stderr, "| %.02f %% [%.01f MB/s]\r",
           100.0*(size?(bytesread/size):1), bytesread/lCopy_time/1048576.);
   fWatch.Continue();
}

//______________________________________________________________________________
Bool_t TFileMerger::Cp(const char *src, const char *dst, Bool_t progressbar,
                       UInt_t buffersize)
{
   // Allows to copy file from src to dst URL.

   Bool_t success = kFALSE;

   TUrl sURL(src, kTRUE);
   TUrl dURL(dst, kTRUE);

   TString oopt = "RECREATE";
   TString ourl = dURL.GetUrl();

   TString raw = "filetype=raw";

   TString opt = sURL.GetOptions();
   if (opt == "")
      opt = raw;
   else
      opt += "&&" + raw;
   sURL.SetOptions(opt);

   opt = dURL.GetOptions();
   if (opt == "")
      opt = raw;
   else
      opt += "&&" + raw;
   dURL.SetOptions(opt);

   char *copybuffer = 0;

   TFile *sfile = 0;
   TFile *dfile = 0;

   sfile = TFile::Open(sURL.GetUrl(), "READ");

   if (!sfile) {
      Error("Cp", "cannot open source file %s", src);
      goto copyout;
   }

   // "RECREATE" does not work always well with XROOTD
   // namely when some pieces of the path are missing;
   // we force "NEW" in such a case
   if (TFile::GetType(ourl, "") == TFile::kNet)
      if (gSystem->AccessPathName(ourl)) {
         oopt = "NEW";
         // Force creation of the missing parts of the path
         opt += "&mkpath=1";
         dURL.SetOptions(opt);
      }

   dfile = TFile::Open(dURL.GetUrl(), oopt);

   if (!dfile) {
      Error("Cp", "cannot open destination file %s", dst);
      goto copyout;
   }

   sfile->Seek(0);
   dfile->Seek(0);

   copybuffer = new char[buffersize];
   if (!copybuffer) {
      Error("Cp", "cannot allocate the copy buffer");
      goto copyout;
   }

   Bool_t   readop;
   Bool_t   writeop;
   Long64_t read;
   Long64_t written;
   Long64_t totalread;
   Long64_t filesize;
   Long64_t b00;
   filesize  = sfile->GetSize();
   totalread = 0;
   fWatch.Start();

   b00 = sfile->GetBytesRead();

   do {
      if (progressbar) PrintProgress(totalread, filesize);

      Long64_t b1 = sfile->GetBytesRead() - b00;

      Long64_t readsize;
      if (filesize - b1 > (Long64_t)buffersize) {
         readsize = buffersize;
      } else {
         readsize = filesize - b1;
      }

      Long64_t b0 = sfile->GetBytesRead();
      sfile->Seek(totalread,TFile::kBeg);
      readop = sfile->ReadBuffer(copybuffer, (Int_t)readsize);
      read   = sfile->GetBytesRead() - b0;
      if (read < 0) {
         Error("Cp", "cannot read from source file %s", src);
         goto copyout;
      }

      Long64_t w0 = dfile->GetBytesWritten();
      writeop = dfile->WriteBuffer(copybuffer, (Int_t)read);
      written = dfile->GetBytesWritten() - w0;
      if (written != read) {
         Error("Cp", "cannot write %d bytes to destination file %s", read, dst);
         goto copyout;
      }
      totalread += read;
   } while (read == (Long64_t)buffersize);

   if (progressbar) {
      PrintProgress(totalread, filesize);
      fprintf(stderr, "\n");
   }

   success = kTRUE;

copyout:
   if (sfile) sfile->Close();
   if (dfile) dfile->Close();

   if (sfile) delete sfile;
   if (dfile) delete dfile;
   if (copybuffer) delete[] copybuffer;

   fWatch.Stop();
   fWatch.Reset();

   return success;
}
//______________________________________________________________________________
void TFileMerger::Reset()
{
   // Reset merger file list.

   fFileList->Clear();
}

//______________________________________________________________________________
Bool_t TFileMerger::AddFile(const char *url)
{
   // Add file to file merger.

   TUUID uuid;
   TString localcopy = "file:/tmp/";
   localcopy += "ROOTMERGE-";
   localcopy += uuid.AsString();
   localcopy += ".root";

   if (!Cp(url, localcopy)) {
      Error("AddFile", "cannot get a local copy of file %s", url);
      return kFALSE;
   }

   TFile *newfile = TFile::Open(localcopy, "READ");
   if (!newfile) {
      Error("AddFile", "cannot open local copy %s of URL %s",
            localcopy.Data(), url);
      return kFALSE;
   } else {
      fFileList->Add(newfile);
      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TFileMerger::OutputFile(const char *outputfile)
{
   // Open merger output file.

   if (fOutputFile)
      delete fOutputFile;

   fOutputFilename = outputfile;

   TUUID uuid;
   TString localcopy = "file:/tmp/";
   localcopy += "ROOTMERGED-";
   localcopy += uuid.AsString();
   localcopy += ".root";

   fOutputFile = TFile::Open(localcopy, "RECREATE");
   fOutputFilename1 = localcopy;

   if (!fOutputFile) {
      Error("OutputFile", "cannot open the MERGER output file %s", localcopy.Data());
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TFileMerger::PrintFiles(Option_t *options)
{
   // Print list of files being merged.

   fFileList->Print(options);
}

//______________________________________________________________________________
Bool_t TFileMerger::Merge()
{
   // Merge the files. If no output file was specified it will write into
   // the file "FileMerger.root" in the working directory. Returns true
   // on success, false in case of error.

   if (!fOutputFile) {
      Info("Merge", "will merge the results to the file "
           "FileMerger.root\nin your working directory, "
           "since you didn't specify a merge filename");
      if (!OutputFile("FileMerger.root")) {
         return kFALSE;
      }
   }

   Bool_t result = MergeRecursive(fOutputFile, fFileList,0);
   if (!result) {
      Error("Merge", "error during merge of your ROOT files");
   } else {
      fOutputFile->Write();
      // copy the result file to the final destination
      Cp(fOutputFilename1, fOutputFilename);
   }

   // remove the temporary result file
   TString path(fOutputFile->GetPath());
   path = path(0, path.Index(':',0));
   gSystem->Unlink(path);
   fOutputFile = 0;

   TIter next(fFileList);
   TFile *file;
   while ((file = (TFile*) next())) {
      // close the files
      file->Close();
      // remove the temporary files
      TString path(file->GetPath());
      path = path(0, path.Index(':',0));
      gSystem->Unlink(path);
   }
   return result;
}

//______________________________________________________________________________
Bool_t TFileMerger::MergeRecursive(TDirectory *target, TList *sourcelist, Int_t isdir)
{
   // Merge all objects in a directory
   // NB. This function is a copy of the hadd function MergeROOTFile
   
   //cout << "Target path: " << target->GetPath() << endl;
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
         } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
            // it's a subdirectory

            //cout << "Found subdirectory " << obj->GetName() << endl;
            // create a new subdir of same name and title in the target file
            target->cd();
            TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

            // newdir is now the starting point of another round of merging
            // newdir still knows its depth within the target file via
            // GetPath(), so we can still figure out where we are in the recursion
            MergeRecursive( newdir, sourcelist,1);

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
               if (!fNoTrees) {
                  globChain->ls();
                  if (fFastMethod) globChain->Merge(target->GetFile(),0,"keep fast");
                  else             globChain->Merge(target->GetFile(),0,"keep");
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
   return kTRUE;
}
