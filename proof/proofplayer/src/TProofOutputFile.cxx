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
TProofOutputFile::TProofOutputFile(const char *path, const char *location, const char *)
                 : TNamed(path,"")
{
   // Main conctructor.
   // The last argument is not used and is kept for compatibility with old versions.

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
      // The directory for the file will be the sandbox unless differently specified
      TString dirPath = gSystem->DirName(fFileName);
      if (dirPath.IsNull() || dirPath == "." || dirPath == "~")
         dirPath = gSystem->WorkingDirectory();
      // Remove prefix, if any
      TString pfx  = gEnv->GetValue("Path.Localroot","");
      if (!pfx.IsNull()) dirPath.Remove(0, pfx.Length());
      // Check if a local data server has been specified
      if (gSystem->Getenv("LOCALDATASERVER")) {
         fDir = gSystem->Getenv("LOCALDATASERVER");
         if (!fDir.EndsWith("/")) fDir += "/";
      }
      fDir += Form("%s", dirPath.Data());
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

   // Copy files locally before merging?
   if (location && !strcmp(location, "LOCAL")) fLocalMerge = kTRUE;
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

   // Include the local data server info, if any
   if (gSystem->Getenv("LOCALDATASERVER")) {
      TString localDS(gSystem->Getenv("LOCALDATASERVER"));
      if (!localDS.EndsWith("/")) localDS += "/";
      fDir.Insert(0, localDS);
   }

   return 0;
}

//______________________________________________________________________________
Long64_t TProofOutputFile::Merge(TCollection* list)
{
   // Merge objects from the list into this object

   // Needs domethign to merge
   if(!list || list->IsEmpty()) return 0;

   TString fileLoc;
   TString outputFileLoc = (fOutputFileName.IsNull()) ? fFileName : fOutputFileName;

   // Get the file merger instance
   TFileMerger *merger = GetFileMerger(fLocalMerge);
   if (!merger) {
      Error("Merge", "could not instantiate the file merger");
      return -1;
   }

   if (!fMerged) {

      merger->OutputFile(outputFileLoc);

      fileLoc = Form("%s/%s", fDir.Data(), GetFileName());
      AddFile(merger, fileLoc);

      fMerged = kTRUE;
   }

   TIter next(list);
   TObject *o = 0;
   while((o = next())) {
      TProofOutputFile *pFile = dynamic_cast<TProofOutputFile *>(o);
      if (pFile) {
         fileLoc = Form("%s/%s", pFile->GetDir(), pFile->GetFileName());
         AddFile(merger, fileLoc);
      }
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
   Info("Print"," merging option:   %s", (fLocalMerge ? "local copy" : "keep remote"));
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
