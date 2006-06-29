// @(#)root/proof:$Name:  $:$Id: TFileMerger.cxx,v 1.8 2006/06/28 20:07:52 brun Exp $
// Author: Andreas Peters + Fons Rademakers   26/5/2005

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
#include "TList.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TUUID.h"
#include "TSystem.h"
#include "TH1.h"
#include "TChain.h"
#include "TKey.h"
#include "TClass.h"

ClassImp(TFileMerger)

//______________________________________________________________________________
TFileMerger::TFileMerger() : fOutputFile(0)
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

   dfile = TFile::Open(dURL.GetUrl(), "RECREATE");

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
      readop = sfile->ReadBuffer(copybuffer, readsize);
      read   = sfile->GetBytesRead() - b0;
      if (read < 0) {
         Error("Cp", "cannot read from source file %s", src);
         goto copyout;
      }

      Long64_t w0 = dfile->GetBytesWritten();
      writeop = dfile->WriteBuffer(copybuffer, read);
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

   Bool_t result = MergeRecursive(fOutputFile, fFileList);
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
Bool_t TFileMerger::MergeRecursive(TDirectory *target, TList *sourcelist)
{
   // Recursively merge objects in the ROOT files.

   TString path(strstr(target->GetPath(), ":"));
   path.Remove(0, 2);

   TFile *first_source = (TFile*)sourcelist->First();
   first_source->cd(path);
   TDirectory *current_sourcedir = gDirectory;

   TChain *globChain = 0;
   TIter nextkey(current_sourcedir->GetListOfKeys());
   TKey *key;
   Bool_t success = kTRUE;

   // gain time, do not add the objects in the list in memory
   TH1::AddDirectory(kFALSE);

   while ((key = (TKey*) nextkey())) {
      first_source->cd(path);
      TObject *obj = key->ReadObj();

      if (obj->IsA()->InheritsFrom("TH1")) {
         Info("MergeRecursive", "merging histogram %s", obj->GetName());
         TH1 *h1 = (TH1*)obj;

         TFile *nextsource = (TFile*)sourcelist->After(first_source);
         while (nextsource) {

            nextsource->cd(path);
            TH1 *h2 = (TH1*)gDirectory->Get(h1->GetName());
            if (h2) {
               h1->Add(h2);
               delete h2;
            }

            nextsource = (TFile*)sourcelist->After(nextsource);
         }
      } else if (obj->IsA()->InheritsFrom("TTree")) {
         Info("MergeRecursive", "merging tree %s", obj->GetName());
         const char *obj_name= obj->GetName();

         globChain = new TChain(obj_name);
         globChain->Add(first_source->GetName());
         TFile *nextsource = (TFile*)sourcelist->After(first_source);
         while (nextsource) {
            globChain->Add(nextsource->GetName());
            nextsource = (TFile*)sourcelist->After(nextsource);
         }

      } else if (obj->IsA()->InheritsFrom("TDirectory")) {
         target->cd();
         TDirectory *newdir = target->mkdir(obj->GetName(), obj->GetTitle());
         if (!MergeRecursive(newdir, sourcelist)) {
            Error("MergeRecursive", "error during merge of directory %s",
                  newdir->GetPath());
            success = kFALSE;
         }
      } else {
         Error("MergeRecursive", "unknown object type, name: %s title: %s",
               obj->GetName(), obj->GetTitle());
         success = kFALSE;
      }

      if (obj) {
         target->cd();

         if (obj->IsA()->InheritsFrom("TTree")) {
            globChain->Merge(target->GetFile() ,0, "keep");
            delete globChain;
         } else
            obj->Write(key->GetName());
      }

   }  // nextkey

   target->Write();

   TH1::AddDirectory(kTRUE);

   return success;
}
