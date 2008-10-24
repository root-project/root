// @(#)root/proof:$Id$
// Author: Long Tran-Thanh   14/09/07

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofOutputFile                                                     //
//                                                                      //
// Small class to steer the merging of files produced on the workers    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofOutputFile.h"
#include <TEnv.h>
#include <TFileMerger.h>
#include <TFile.h>
#include <TList.h>
#include <TObjArray.h>
#include <TObject.h>
#include <TObjString.h>
#include <TProofServ.h>
#include <TSystem.h>
#include <TUUID.h>

ClassImp(TProofOutputFile)

//________________________________________________________________________________
TProofOutputFile::TProofOutputFile(const char* path,
                                   const char* location, const char* mode)
                 : TNamed(path,"")
{
   // Main conctructor

   fMerged = kFALSE;
   fMerger = 0;

   TUrl u(path, kTRUE);
   // File name
   fFileName = u.GetFile();
   // Unique file name
   fFileName1 = GetTmpName(fFileName.Data());
   // Path
   fIsLocal = kFALSE;
   fDir = u.GetUrl();
   Int_t pos = fDir.Index(fFileName);
   if (pos != kNPOS)
      fDir.Remove(pos);

   if (fDir == "file:") {
      fIsLocal = kTRUE;
      TString dirPath = gSystem->WorkingDirectory();
      if (gSystem->Getenv("XRDCF")) {
         // The directory for the file will be the sandbox
         TString pfx  = gEnv->GetValue("Path.Localroot","");
         fDir = Form("root://%s",gSystem->HostName());
         if (gSystem->Getenv("XRDPORT")) {
            TString sp(gSystem->Getenv("XRDPORT"));
            if (sp.IsDigit())
               fDir += Form(":%s", sp.Data());
         }
         if (!pfx.IsNull())
            dirPath.Remove(0, pfx.Length());
      } else {
         fDir += "/";
      }
      fDir += Form("/%s", dirPath.Data());
   }
   // Notify
   Info("TProofOutputFile", "dir: %s", fDir.Data());

   // Default output file name
   fOutputFileName = gEnv->GetValue("Proof.OutputFile", "");
   if (!fOutputFileName.IsNull() && !fOutputFileName.EndsWith("/"))
      fOutputFileName += "/";
   // Add default file name
   fOutputFileName += path;
   if (!fOutputFileName.EndsWith(".root"))
      fOutputFileName += ".root";
   // Resolve placeholders
   ResolveKeywords(fOutputFileName);
   Info("TProofOutputFile", "output file url: %s", fOutputFileName.Data());

   // Location
   fLocation = "REMOTE";
   if (location && strlen(location) > 0) {
      fLocation = location;
      if (fLocation.CompareTo("LOCAL", TString::kIgnoreCase) &&
          fLocation.CompareTo("REMOTE", TString::kIgnoreCase)) {
         Warning("TProofOutputFile","unknown location %s: ignore (use: \"REMOTE\")", location);
         fLocation = "REMOTE";
      }
      fLocation.ToUpper();
   }
   // Mode
   fMode = "CENTRAL";
   if (mode && strlen(mode) > 0) {
      fMode = mode;
      if (fMode.CompareTo("CENTRAL", TString::kIgnoreCase) &&
          fMode.CompareTo("SEQUENTIAL", TString::kIgnoreCase)) {
         Warning("TProofOutputFile","unknown mode %s: ignore (use: \"CENTRAL\")", mode);
         fMode = "CENTRAL";
      }
      fMode.ToUpper();
   }
}

//________________________________________________________________________________
TProofOutputFile::~TProofOutputFile()
{
   // Main destructor

   if (fMerger) delete fMerger;
}

//______________________________________________________________________________
TString TProofOutputFile::GetTmpName(const char* name)
{
   // Create a temporary unique name for this file

   TUUID uuid;

   TString tmpName(name);
   Ssiz_t pos = tmpName.Last('.');
   if (pos != kNPOS)
      tmpName.Insert(pos,Form("-%s",uuid.AsString()));
   else
      tmpName += Form("-%s",uuid.AsString());

   // Done
   return tmpName;
}

//______________________________________________________________________________
void TProofOutputFile::SetFileName(const char* name)
{
   // Set the file name

   fFileName = name;
   fFileName1 = GetTmpName(name);
}

//______________________________________________________________________________
void TProofOutputFile::SetOutputFileName(const char *name)
{
   // Set the name of the output file; in the form of an Url.

   if (name && strlen(name) > 0) {
      fOutputFileName = name;
      ResolveKeywords(fOutputFileName);
      Info("SetOutputFileName", "output file url: %s", fOutputFileName.Data());
   } else {
      fOutputFileName = "";
   }
}

//______________________________________________________________________________
void TProofOutputFile::ResolveKeywords(TString &fname)
{
   // Replace <user> and <group> placeholders in fname

   // Replace <user>, if any
   if (fname.Contains("<user>")) {
      TString user = "nouser";
      // Get user logon name
      UserGroup_t *pw = gSystem->GetUserInfo();
      if (pw) {
         user = pw->fUser;
         delete pw;
      }
      fname.ReplaceAll("<user>", user);
   }
   // Replace <group>, if any
   if (fname.Contains("<group>")) {
      if (gProofServ && gProofServ->GetGroup() && strlen(gProofServ->GetGroup()))
         fname.ReplaceAll("<group>", gProofServ->GetGroup());
      else
         fname.ReplaceAll("<group>", "default");
   }
}

//______________________________________________________________________________
TFile* TProofOutputFile::OpenFile(const char* opt)
{
   // Open the file using the unique temporary name

   if (fFileName1.IsNull())
      return 0;

   // Create the path
   TString fileLoc = (fIsLocal || fDir.IsNull()) ? fFileName1
                                : Form("%s/%s", fDir.Data(), fFileName1.Data());
   // Open the file
   TFile *retFile = TFile::Open(fileLoc, opt);

   return retFile;
}

//______________________________________________________________________________
Int_t TProofOutputFile::AdoptFile(TFile *f)
{
   // Adopt a file already open.
   // Return 0 if OK, -1 in case of failure

   if (!f || f->IsZombie())
      return -1;

   // Set the name and dir
   TUrl u(*(f->GetEndpointUrl()));
   fIsLocal = kFALSE;
   if (!strcmp(u.GetProtocol(), "file")) {
      fIsLocal = kTRUE;
      fDir = u.GetFile();
   } else {
      fDir = u.GetUrl();
   }
   fFileName1 = gSystem->BaseName(fDir.Data());
   fFileName = fFileName1;
   fDir.ReplaceAll(fFileName1, "");

   return 0;
}

//______________________________________________________________________________
Long64_t TProofOutputFile::Merge(TCollection* list)
{
   // Merge objects from the list into this object

   if(!list || list->IsEmpty())
      return 0; 

   TString fileLoc;
   TString outputFileLoc = (fOutputFileName.IsNull()) ? fFileName : fOutputFileName;

   if (fMode == "SEQUENTIAL") {
      TFileMerger* merger = new TFileMerger;
      if (fLocation == "LOCAL") {
         merger->OutputFile(outputFileLoc);
         if (!fMerged) {
            fileLoc = Form("%s/%s", fDir.Data(), GetFileName());
            AddFile(merger, fileLoc);
            Unlink(outputFileLoc);
         } else {
            AddFile(merger, outputFileLoc);
            Unlink(outputFileLoc);
         }

         TList* elist = new TList;
         elist->AddAll(list); 
         TIter next(elist);
         TProofOutputFile* pFile = 0;

         while ((pFile = (TProofOutputFile*)next())) {
            fileLoc = Form("%s/%s", pFile->GetDir(), pFile->GetFileName());
            AddFile(merger, fileLoc);
         }

         Bool_t result = merger->Merge();
         if (!result) {
            NotifyError("TProofOutputFile::Merge: error from TFileMerger::Merge()");
            return -1;
         }

         if (!fMerged) {
            fileLoc = Form("%s/%s", fDir.Data(), GetFileName());
            Unlink(fileLoc);
            fMerged = kTRUE;
         }

         next.Reset();
         while ((pFile = (TProofOutputFile*)next())) {
            fileLoc = Form("%s/%s", pFile->GetDir(), pFile->GetFileName());
            Unlink(fileLoc);
         }
      } else if (fLocation == "REMOTE") {

         TString outputFileLoc2 = GetTmpName(fOutputFileName);
         TString tmpOutputLoc = (outputFileLoc.BeginsWith("root://")) ? GetTmpName(fFileName) : "";
         TList* fileList = new TList;

         if (!fMerged) {
            fileLoc = Form("%s/%s", fDir.Data(), GetFileName());
            TFile* fCurrFile = TFile::Open(fileLoc,"READ");
            if (!fCurrFile) {
               Warning("Merge","Cannot open file: %s", fileLoc.Data());
            } else {
               fileList->Add(fCurrFile);
               Info("Merge", "now adding file :%s\n", fCurrFile->GetPath());
            }
            Unlink(outputFileLoc);
         } else {
            if (tmpOutputLoc.IsNull()) {
               gSystem->Rename(outputFileLoc,outputFileLoc2);
            } else {
               TFile::Cp(outputFileLoc, outputFileLoc2);
               Unlink(outputFileLoc);
            }

            TFile* fCurrOutputFile = TFile::Open(outputFileLoc2,"READ");
            if (!fCurrOutputFile) {
               Warning("Merge","Cannot open tmp output file: %s", outputFileLoc2.Data());
            } else {
               fileList->Add(fCurrOutputFile);
            }
         }

         TList* elist = new TList;
         elist->AddAll(list);
         TIter next(elist);
         TProofOutputFile* pFile = 0;

         while ((pFile = (TProofOutputFile*)next())) {
            fileLoc = Form("%s/%s", pFile->GetDir(), pFile->GetFileName());

            TFile* fCurrFile = TFile::Open(fileLoc.Data(),"READ");
            if (!fCurrFile) {
               Warning("Merge","Cannot open file: %s", fileLoc.Data());
               continue;
            } else {
               fileList->Add(fCurrFile);
            }
         }

         TFile* outputFile;
         if (tmpOutputLoc.IsNull()) {
            outputFile = TFile::Open(outputFileLoc, "RECREATE");
         } else {
            outputFile = TFile::Open(tmpOutputLoc,"RECREATE");
         }

         if (!outputFile) {
            Error("Merge","cannot open output file %s",outputFileLoc.Data());
            return -1;
         }
         Bool_t result =  merger->MergeRecursive(outputFile, fileList, 0);
         if (!result) {
            NotifyError("TProofOutputFile::Merge: error from TFileMerger::MergeRecursive()");

            TIter fnext(fileList);
            TFile *fCurrFile = 0;
            while ((fCurrFile = (TFile*)fnext())) {
               fCurrFile->Close();
            }
            return -1;
         } else {
            outputFile->Write();
            outputFile->Close();

            TIter fnext(fileList);
            TFile *fCurrFile = 0;
            while ((fCurrFile = (TFile*)fnext())) {
               fCurrFile->Close();
            }

            if (!fMerged) {
               fileLoc = Form("%s/%s", fDir.Data(), GetFileName());
               Unlink(fileLoc);
               fMerged = kTRUE;
            }

            next.Reset();
            while ((pFile = (TProofOutputFile *)next())) {
               fileLoc = Form("%s/%s", pFile->GetDir(), pFile->GetFileName());
               Unlink(fileLoc);
            }

            Unlink(outputFileLoc2); 
            if (!tmpOutputLoc.IsNull()) {
               TFile::Cp(tmpOutputLoc,outputFileLoc);
               Unlink(tmpOutputLoc);
            }
         } //end else
      } else {   // end fLocation = "Remote"
         // the given merging location is not valid
         Error("Merge", "invalid location value: %s", fLocation.Data());
         return -1;
      }
      SafeDelete(merger);

      // end fMode = "SEQUENTIAL"

   } else if (fMode == "CENTRAL") {

      // if we merge the outputfiles centrally

      if (fLocation != "REMOTE" && fLocation != "LOCAL") {
         Error("Merge", "invalid location value: %s", fLocation.Data());
         return -1;
      }

      // Get the file merger instance
      Bool_t isLocal = (fLocation == "REMOTE") ? kFALSE : kTRUE;
      TFileMerger *merger = GetFileMerger(isLocal);
      if (!merger) {
         Error("Merge", "could not instantiate the file merger");
         return -1;
      }

      if (!fMerged) {

         merger->OutputFile(outputFileLoc);
         Unlink(outputFileLoc);

         fileLoc = Form("%s/%s", fDir.Data(), GetFileName());
         AddFile(merger, fileLoc);

         fMerged = kTRUE;
      }

      TList* elist = new TList;
      elist->AddAll(list); 
      TIter next(elist);
      TProofOutputFile* pFile = 0;

      while((pFile = (TProofOutputFile*)next())) {
         fileLoc = Form("%s/%s", pFile->GetDir(), pFile->GetFileName());
         AddFile(merger, fileLoc);
      }

      // end fMode = "CENTRAL"

   } else {
      Error("Merge", "invalid mode value: %s", fMode.Data());
      return -1;
   }

   // Done
   return 0;
}

//______________________________________________________________________________
void TProofOutputFile::Print(Option_t *) const
{
   // Dump the class content

   Info("Print","-------------- %s : start ------------", GetName());
   Info("Print"," dir:              %s", fDir.Data());
   Info("Print"," file name:        %s", fFileName.Data());
   Info("Print"," location:         %s", fLocation.Data());
   Info("Print"," mode:             %s", fMode.Data());
   Info("Print"," output file name: %s", fOutputFileName.Data());
   Info("Print"," ordinal:          %s", fWorkerOrdinal.Data());
   Info("Print","-------------- %s : done -------------", GetName());

   return;
}

//______________________________________________________________________________
void TProofOutputFile::NotifyError(const char *msg)
{
   // Notify error message

   if (msg) {
      if (gProofServ)
         gProofServ->SendAsynMessage(msg);
      else
         Printf(msg);
   } else {
      Info("NotifyError","called with empty message");
   }

   return;
}

//______________________________________________________________________________
void TProofOutputFile::AddFile(TFileMerger *merger, const char *path)
{
   // Add file to merger, checking the result

   if (merger && path) {
      if (!merger->AddFile(path))
         NotifyError(Form("TProofOutputFile::AddFile:"
                          " error from TFileMerger::AddFile(%s)", path));
   }
}

//______________________________________________________________________________
void TProofOutputFile::Unlink(const char *path)
{
   // Unlink path

   if (path) {
      if (!gSystem->AccessPathName(path)) {
         if (gSystem->Unlink(path) != 0)
            NotifyError(Form("TProofOutputFile::Unlink:"
                             " error from TSystem::Unlink(%s)", path));
      }
   }
}

//______________________________________________________________________________
TFileMerger *TProofOutputFile::GetFileMerger(Bool_t local)
{
   // Get instance of the file merger to be used in "CENTRAL" mode

   if (!fMerger)
      fMerger = new TFileMerger(local);
   return fMerger;
}
