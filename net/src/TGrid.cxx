// @(#)root/net:$Name:  $:$Id: TGrid.cxx,v 1.9 2005/05/20 09:59:35 rdm Exp $
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGrid                                                                //
//                                                                      //
// Abstract base class defining interface to common GRID services.      //
//                                                                      //
// To open a connection to a GRID use the static method Connect().      //
// The argument of Connect() is of the form:                            //
//    <grid>[://<host>][:<port>], e.g.                                  //
// alien, alien://alice.cern.ch, globus://glsvr1.cern.ch, ...           //
// Depending on the <grid> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TGridResult.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGrid.h"
#include "TUrl.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TFile.h"
#include "TList.h"
#include "TError.h"
#include "TUUID.h"
#include "TSystem.h"
#include "TH1.h"
#include "TChain.h"
#include "TKey.h"

TGrid *gGrid = 0;


ClassImp(TGrid)


//______________________________________________________________________________
TGrid::TGrid() : fPort(-1), fMergerOutputFile(0)
{
   // Create grid interface object.

   fMergerFileList = new TList;
   fMergerFileList->SetOwner(kTRUE);
}

//______________________________________________________________________________
TGrid *TGrid::Connect(const char *grid, const char *uid, const char *pw,
                      const char *options)
{
   // The grid should be of the form:  <grid>://<host>[:<port>],
   // e.g.:  alien://alice.cern.ch, globus://glsrv1.cern.ch, ...
   // The uid is the username and pw the password that should be used for
   // the connection. Depending on the <grid> the shared library (plugin)
   // for the selected system will be loaded. When the connection could not
   // be opened 0 is returned. For AliEn the supported options are:
   // -domain=<domain name>
   // -debug=<debug level from 1 to 10>
   // Example: "-domain=cern.ch -debug=5"

   TPluginHandler *h;
   TGrid *g = 0;

   if (!grid) {
      ::Error("TGrid::Connect", "no grid specified");
      return 0;
   }
   if (!uid)
      uid = "";
   if (!pw)
      pw = "";
   if (!options)
      options = "";

   if ((h = gROOT->GetPluginManager()->FindHandler("TGrid", grid))) {
      if (h->LoadPlugin() == -1)
         return 0;
      g = (TGrid *) h->ExecPlugin(4, grid, uid, pw, options);
   }

   return g;
}

//______________________________________________________________________________
TGrid::~TGrid()
{
   // Cleanup.

  if (fMergerFileList)
    delete fMergerFileList;

  if (fMergerOutputFile)
    delete fMergerOutputFile;
}

//______________________________________________________________________________
void TGrid::PrintProgress(Long64_t bytesread, Long64_t size)
{
   // Print file copy progress.

   fprintf(stderr, "[TGrid::Cp] Total %.02f MB\t|", (Double_t)size/1048576);
   for (int l = 0; l < 20; l++) {
      if (l < 20*bytesread/size)
         fprintf(stderr, "=");
      if (l == 20*bytesread/size)
         fprintf(stderr, ">");
      if (l > 20*bytesread/size)
         fprintf(stderr, ".");
   }

   fWatch.Stop();
   Double_t lCopy_time = fWatch.RealTime();
   fprintf(stderr, "| %.02f %% [%.01f MB/s]\r",
          100.0*bytesread/size, bytesread/lCopy_time/1048576.);
   fWatch.Continue();
}

//______________________________________________________________________________
Bool_t TGrid::Cp(const char *src, const char *dst, Bool_t progressbar,
                 UInt_t buffersize)
{
   // Allows to copy file from src to dst URL.

   Bool_t success = kFALSE;

   TUrl sURL(src, kTRUE);
   TUrl dURL(dst, kTRUE);

   char *copybuffer = 0;

   TFile *sfile = 0;
   TFile *dfile = 0;

   sfile = TFile::Open(src, "-READ");

   if (!sfile) {
      Error("Cp", "cannot open source file %s", src);
      goto copyout;
   }

   dfile = TFile::Open(dst, "-RECREATE");

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

   b00 = (Long64_t)sfile->GetBytesRead();

   do {
      if (progressbar) PrintProgress(totalread, filesize);

      Long64_t b1 = (Long64_t)sfile->GetBytesRead() - b00;

      Long64_t readsize;
      if (filesize - b1 > (Long64_t)buffersize) {
         readsize = buffersize;
      } else {
         readsize = filesize - b1;
      }

      Long64_t b0 = (Long64_t)sfile->GetBytesRead();
      readop = sfile->ReadBuffer(copybuffer, readsize);
      read   = (Long64_t)sfile->GetBytesRead() - b0;
      if (read < 0) {
         Error("Cp", "cannot read from source file %s", src);
         goto copyout;
      }

      Long64_t w0 = (Long64_t)dfile->GetBytesWritten();
      writeop = dfile->WriteBuffer(copybuffer, read);
      written = (Long64_t)dfile->GetBytesWritten() - w0;
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
   if (copybuffer) delete copybuffer;

   fWatch.Stop();
   fWatch.Reset();

   return success;
}

//______________________________________________________________________________
void TGrid::MergerReset()
{
   // Reset merger file list.

   fMergerFileList->Clear();
}

//______________________________________________________________________________
Bool_t TGrid::MergerAddFile(const char *url)
{
   // Add file to file merger.

   TUUID uuid;
   TString localcopy = "file:/tmp/";
   localcopy += "ROOTMERGE-";
   localcopy += uuid.AsString();
   localcopy += ".root";

   if (!Cp(url, localcopy)) {
      Error("MergerAddFile", "cannot get a local copy of file %s", url);
      return kFALSE;
   }

   TFile *newfile = TFile::Open(localcopy, "READ");
   if (!newfile) {
      Error("MergerAddFile", "cannot open local copy %s of URL %s",
            localcopy.Data(), url);
      return kFALSE;
   } else {
      fMergerFileList->Add(newfile);
      return  kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TGrid::MergerOutputFile(const char *outputfile)
{
   // Open merger output file.

   if (fMergerOutputFile)
      delete fMergerOutputFile;

   fMergerOutputFilename = outputfile;

   TUUID uuid;
   TString localcopy = "file:/tmp/";
   localcopy += "ROOTMERGED-";
   localcopy += uuid.AsString();
   localcopy += ".root";

   fMergerOutputFile = TFile::Open(localcopy, "RECREATE");
   fMergerOutputFilename1 = localcopy;

   if (!fMergerOutputFile) {
      Error("MergerOutputFile", "cannot open the MERGER outputfile %s", localcopy.Data());
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGrid::MergerPrintFiles(Option_t *options)
{
   // Print list of files being merged.

   fMergerFileList->Print(options);
}

//______________________________________________________________________________
Bool_t TGrid::MergerMerge()
{
   // Merge the files.

   if (!fMergerOutputFile) {
      Info("MergerMerge", "will merge the results to the file "
           "GridMergerMerged.root in your working directory, "
           "since you didn't specify a merge filename");
      if (!MergerOutputFile("GridMergerMerged.root")) {
         return kFALSE;
      }
   }

   Bool_t result = MergerMergeRecursive(fMergerOutputFile, fMergerFileList);
   if (!result) {
      Error("MergerMerge", "error during merge of your ROOT files");
   } else {
      fMergerOutputFile->Write();
      // copy the result file to the final destination
      Cp(fMergerOutputFilename1, fMergerOutputFilename);
   }

   // remove the temporary result file
   TString path(fMergerOutputFile->GetPath());
   path = path(0, path.Index(':',0));
   gSystem->Unlink(path);
   fMergerOutputFile = 0;

   TIter next(fMergerFileList);
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
Bool_t TGrid::MergerMergeRecursive(TDirectory *target, TList *sourcelist)
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
         Info("MergerMergeRecursive", "merging histogram %s", obj->GetName());
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
         Info("MergerMergeRecursive", "merging tree %s", obj->GetName());
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
         if (!MergerMergeRecursive(newdir, sourcelist)) {
            Error("MergerMergeRecursive", "error during merge of directory %s",
                  newdir->GetPath());
            success = kFALSE;
         }
      } else {
         Error("MergerMergeRecursive", "unknown object type, name: %s title: %s",
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
