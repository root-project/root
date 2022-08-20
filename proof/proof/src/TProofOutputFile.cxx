// @(#)root/proof:$Id$
// Author: Long Tran-Thanh   14/09/07

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TProofOutputFile
\ingroup proofkernel

Class to steer the merging of files produced on the workers

*/

#include "TProofOutputFile.h"
#include <TEnv.h>
#include <TError.h>
#include <TFileCollection.h>
#include <TFileInfo.h>
#include <TFileMerger.h>
#include <TFile.h>
#include <TList.h>
#include <TObjArray.h>
#include <TObject.h>
#include <TObjString.h>
#include <TProofDebug.h>
#include <TProofServ.h>
#include <TSystem.h>
#include <TUUID.h>

ClassImp(TProofOutputFile);

////////////////////////////////////////////////////////////////////////////////
/// Main constructor

TProofOutputFile::TProofOutputFile(const char *path,
                                   ERunType type, UInt_t opt, const char *dsname)
                 : TNamed(path, ""), fRunType(type), fTypeOpt(opt)
{
   fIsLocal = kFALSE;
   fMerged = kFALSE;
   fMerger = 0;
   fDataSet = 0;
   ResetBit(TProofOutputFile::kRetrieve);
   ResetBit(TProofOutputFile::kSwapFile);

   Init(path, dsname);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with the old signature, kept for convenience and backard compatibility.
/// Options:
///             'M'      merge: finally merge the created files
///             'L'      local: copy locally the files before merging (implies 'M')
///             'D'      dataset: create a TFileCollection
///             'R'      register: dataset run with dataset registration
///             'O'      overwrite: force dataset replacement during registration
///             'V'      verify: verify the registered dataset
///             'H'      merge histograms in one go (option to TFileMerger)
/// Special 'option' values for backward compatibility:
///              ""      equivalent to "M"
///         "LOCAL"      equivalent to "ML" or "L"

TProofOutputFile::TProofOutputFile(const char *path,
                                   const char *option, const char *dsname)
                 : TNamed(path, "")
{
   fIsLocal = kFALSE;
   fMerged = kFALSE;
   fMerger = 0;
   fDataSet = 0;
   fMergeHistosOneGo = kFALSE;

   // Fill the run type and option type
   fRunType = kMerge;
   fTypeOpt = kRemote;
   if (option && strlen(option) > 0) {
      TString opt(option);
      if (opt.Contains("L") || (opt == "LOCAL")) fTypeOpt = kLocal;
      if (opt.Contains("H")) fMergeHistosOneGo = kTRUE;
      if (!opt.Contains("M") && opt.Contains("D")) {
         // Dataset creation mode
         fRunType = kDataset;
         fTypeOpt = kCreate;
         if (opt.Contains("R")) fTypeOpt = (ETypeOpt) (fTypeOpt | kRegister);
         if (opt.Contains("O")) fTypeOpt = (ETypeOpt) (fTypeOpt | kOverwrite);
         if (opt.Contains("V")) fTypeOpt = (ETypeOpt) (fTypeOpt | kVerify);
      }
   }

   Init(path, dsname);
}

////////////////////////////////////////////////////////////////////////////////
/// Initializer. Called by all constructors

void TProofOutputFile::Init(const char *path, const char *dsname)
{
   fLocalHost = TUrl(gSystem->HostName()).GetHostFQDN();
   Int_t port = gEnv->GetValue("ProofServ.XpdPort", -1);
   if (port > -1) {
      fLocalHost += ":";
      fLocalHost += port;
   }

   TString xpath(path);
   // Resolve the relevant placeholders in fFileName (e.g. root://a.ser.ver//data/dir/<group>/<user>/file)
   TProofServ::ResolveKeywords(xpath, 0);
   TUrl u(xpath, kTRUE);
   // File name
   fFileName = u.GetFile();
   // The name is used to identify this entity
   SetName(gSystem->BaseName(fFileName.Data()));
   // The title is the dataset name in the case such option is chosen.
   // In the merging case it can be the final location of the file on the client if the retrieve
   // option is chosen; if the case, this set in TProofPlayer::MergeOutputFiles.
   if (fRunType == kDataset) {
      if (dsname && strlen(dsname) > 0) {
         // This is the dataset name in case such option is chosen
         SetTitle(dsname);
      } else {
         // Default dataset name
         SetTitle(GetName());
      }
   }
   // Options and anchor, if any
   if (u.GetOptions() && strlen(u.GetOptions()) > 0)
      fOptionsAnchor += TString::Format("?%s", u.GetOptions());
   if (u.GetAnchor() && strlen(u.GetAnchor()) > 0)
      fOptionsAnchor += TString::Format("#%s", u.GetAnchor());
   // Path
   fIsLocal = kFALSE;
   fDir = u.GetUrl();
   Int_t pos = fDir.Index(fFileName);
   if (pos != kNPOS) fDir.Remove(pos);
   fRawDir = fDir;

   if (fDir.BeginsWith("file:")) {
      fIsLocal = kTRUE;
      // For local files, the user is allowed to create files under the specified directory.
      // If this is not the case, the file is rooted automatically to the assigned dir which
      // is the datadir for dataset creation runs, and the working dir for merging runs
      TString dirPath = gSystem->GetDirName(fFileName);
      fFileName = gSystem->BaseName(fFileName);
      if (AssertDir(dirPath) != 0)
         Error("Init", "problems asserting path '%s'", dirPath.Data());
      TString dirData = (!IsMerge() && gProofServ) ? gProofServ->GetDataDir()
                                                   : gSystem->WorkingDirectory();
      if ((dirPath[0] == '/') && gSystem->AccessPathName(dirPath, kWritePermission)) {
         Warning("Init", "not allowed to create files under '%s' - chrooting to '%s'",
                         dirPath.Data(), dirData.Data());
         dirPath.Insert(0, dirData);
      } else if (dirPath.BeginsWith("..")) {
         dirPath.Remove(0, 2);
         if (dirPath[0] != '/') dirPath.Insert(0, "/");
         dirPath.Insert(0, dirData);
      } else if (dirPath[0] == '.' || dirPath[0] == '~') {
         dirPath.Remove(0, 1);
         if (dirPath[0] != '/') dirPath.Insert(0, "/");
         dirPath.Insert(0, dirData);
      } else if (dirPath.IsNull()) {
         dirPath = dirData;
      }
      // Make sure that session-tag, ordinal and query sequential number are present otherwise
      // we may override outputs from other workers
      if (gProofServ) {
         if (!IsMerge() || (!dirPath.BeginsWith(gProofServ->GetDataDir()) &&
                            !dirPath.BeginsWith(gSystem->WorkingDirectory()))) {
            if (!dirPath.Contains(gProofServ->GetOrdinal())) {
               if (!dirPath.EndsWith("/")) dirPath += "/";
               dirPath += gProofServ->GetOrdinal();
            }
         }
         if (!IsMerge()) {
            if (!dirPath.Contains(gProofServ->GetSessionTag())) {
               if (!dirPath.EndsWith("/")) dirPath += "/";
               dirPath += gProofServ->GetSessionTag();
            }
            if (!dirPath.Contains("<qnum>")) {
               if (!dirPath.EndsWith("/")) dirPath += "/";
               dirPath += "<qnum>";
            }
            // Resolve the relevant placeholders
            TProofServ::ResolveKeywords(dirPath, 0);
         }
      }
      // Save the raw directory
      fRawDir = dirPath;
      // Make sure the path exists
      if (AssertDir(dirPath) != 0)
         Error("Init", "problems asserting path '%s'", dirPath.Data());
      // Take into account local server settings
      TProofServ::GetLocalServer(fDir);
      TProofServ::FilterLocalroot(dirPath, fDir);
      // The path to be used to address the file
      fDir += dirPath;
   }
   // Notify
   Info("Init", "dir: %s (raw: %s)", fDir.Data(), fRawDir.Data());

   // Default output file name
   ResetBit(TProofOutputFile::kOutputFileNameSet);
   fOutputFileName = "<file>";
   if (gEnv->Lookup("Proof.OutputFile")) {
      fOutputFileName = gEnv->GetValue("Proof.OutputFile", "<file>");
      SetBit(TProofOutputFile::kOutputFileNameSet);
   }
   // Add default file name
   TString fileName = path;
   if (!fileName.EndsWith(".root")) fileName += ".root";
   // Make sure that the file name was inserted (may not happen if the placeholder <file> is missing)
   if (!fOutputFileName.IsNull() && !fOutputFileName.Contains("<file>")) {
      if (!fOutputFileName.EndsWith("/")) fOutputFileName += "/";
         fOutputFileName += fileName;
   }
   // Resolve placeholders
   fileName.ReplaceAll("<ord>",""); // No ordinal in the final merged file
   TProofServ::ResolveKeywords(fOutputFileName, fileName);
   Info("Init", "output file url: %s", fOutputFileName.Data());
   // Fill ordinal
   fWorkerOrdinal = "<ord>";
   TProofServ::ResolveKeywords(fWorkerOrdinal, 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Main destructor

TProofOutputFile::~TProofOutputFile()
{
   if (fDataSet) delete fDataSet;
   if (fMerger) delete fMerger;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the name of the output file; in the form of an Url.

void TProofOutputFile::SetOutputFileName(const char *name)
{
   if (name && strlen(name) > 0) {
      fOutputFileName = name;
      TProofServ::ResolveKeywords(fOutputFileName);
      PDB(kOutput,1) Info("SetOutputFileName", "output file url: %s", fOutputFileName.Data());
   } else {
      fOutputFileName = "";
   }
   SetBit(TProofOutputFile::kOutputFileNameSet);
}

////////////////////////////////////////////////////////////////////////////////
/// Open the file using the unique temporary name

TFile* TProofOutputFile::OpenFile(const char* opt)
{
   if (fFileName.IsNull()) return 0;

   // Create the path
   TString fileLoc;
   fileLoc.Form("%s/%s%s", fRawDir.Data(), fFileName.Data(), fOptionsAnchor.Data());

   // Open the file
   TFile *retFile = TFile::Open(fileLoc, opt);

   return retFile;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt a file already open.
/// Return 0 if OK, -1 in case of failure

Int_t TProofOutputFile::AdoptFile(TFile *f)
{
   if (!f || (f && f->IsZombie())) {
      Error("AdoptFile", "file is undefined or zombie!");
      return -1;
   }
   const TUrl *u = f->GetEndpointUrl();
   if (!u) {
      Error("AdoptFile", "file end-point url is undefined!");
      return -1;
   }

   // Set the name and dir
   fIsLocal = kFALSE;
   if (!strcmp(u->GetProtocol(), "file")) {
      fIsLocal = kTRUE;
      fDir = u->GetFile();
   } else {
      fDir = u->GetUrl();
   }
   fFileName = gSystem->BaseName(fDir.Data());
   fDir.ReplaceAll(fFileName, "");
   fRawDir = fDir;

   // If local remove prefix, if any
   if (fIsLocal) {
      TString localDS;
      TProofServ::GetLocalServer(localDS);
      if (!localDS.IsNull()) {
         TProofServ::FilterLocalroot(fDir, localDS);
         fDir.Insert(0, localDS);
      }
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Merge objects from the list into this object

Long64_t TProofOutputFile::Merge(TCollection* list)
{
   PDB(kOutput,2) Info("Merge","enter: merge? %d", IsMerge());

   // Needs somethign to merge
   if(!list || list->IsEmpty()) return 0;

   if (IsMerge()) {
      // Build-up the merger
      TString fileLoc;
      TString outputFileLoc = (fOutputFileName.IsNull()) ? fFileName : fOutputFileName;
      // Get the file merger instance
      Bool_t localMerge = (fRunType == kMerge && fTypeOpt == kLocal) ? kTRUE : kFALSE;
      TFileMerger *merger = GetFileMerger(localMerge);
      if (!merger) {
         Error("Merge", "could not instantiate the file merger");
         return -1;
      }

      if (!fMerged) {
         merger->OutputFile(outputFileLoc);
         fileLoc.Form("%s/%s", fDir.Data(), GetFileName());
         AddFile(merger, fileLoc);
         fMerged = kTRUE;
      }

      TIter next(list);
      TObject *o = 0;
      while((o = next())) {
         TProofOutputFile *pFile = dynamic_cast<TProofOutputFile *>(o);
         if (pFile) {
            fileLoc.Form("%s/%s", pFile->GetDir(), pFile->GetFileName());
            AddFile(merger, fileLoc);
         }
      }
   } else {
      // Get the reference MSS url, if any
      TUrl mssUrl(gEnv->GetValue("ProofServ.PoolUrl",""));
      // Build-up the TFileCollection
      TFileCollection *dataset = GetFileCollection();
      if (!dataset) {
         Error("Merge", "could not instantiate the file collection");
         return -1;
      }
      fMerged = kTRUE;
      TString path;
      TFileInfo *fi = 0;
      // If new, add ourseelves
      dataset->Update();
      PDB(kOutput,2) Info("Merge","dataset: %s (nfiles: %lld)", dataset->GetName(), dataset->GetNFiles());
      if (dataset->GetNFiles() == 0) {
         // Save the export and raw urls
         path.Form("%s/%s%s", GetDir(), GetFileName(), GetOptionsAnchor());
         fi = new TFileInfo(path);
         // Add also an URL with the redirector path, if any
         if (mssUrl.IsValid()) {
            TUrl ur(fi->GetFirstUrl()->GetUrl());
            ur.SetProtocol(mssUrl.GetProtocol());
            ur.SetHost(mssUrl.GetHost());
            ur.SetPort(mssUrl.GetPort());
            if (mssUrl.GetUser() && strlen(mssUrl.GetUser()) > 0)
               ur.SetUser(mssUrl.GetUser());
            fi->AddUrl(ur.GetUrl());
         }
         // Add special local URL to keep track of the file
         path.Form("%s/%s?node=%s", GetDir(kTRUE), GetFileName(), GetLocalHost());
         fi->AddUrl(path);
         PDB(kOutput,2) fi->Print();
         // Now add to the dataset
         dataset->Add(fi);
      }

      TIter next(list);
      TObject *o = 0;
      while((o = next())) {
         TProofOutputFile *pFile = dynamic_cast<TProofOutputFile *>(o);
         if (pFile) {
            // Save the export and raw urls
            path.Form("%s/%s%s", pFile->GetDir(), pFile->GetFileName(), pFile->GetOptionsAnchor());
            fi = new TFileInfo(path);
            // Add also an URL with the redirector path, if any
            if (mssUrl.IsValid()) {
               TUrl ur(fi->GetFirstUrl()->GetUrl());
               ur.SetProtocol(mssUrl.GetProtocol());
               ur.SetHost(mssUrl.GetHost());
               ur.SetPort(mssUrl.GetPort());
               if (mssUrl.GetUser() && strlen(mssUrl.GetUser()) > 0)
                  ur.SetUser(mssUrl.GetUser());
               fi->AddUrl(ur.GetUrl());
            }
            // Add special local URL to keep track of the file
            path.Form("%s/%s?node=%s", pFile->GetDir(kTRUE), pFile->GetFileName(), pFile->GetLocalHost());
            fi->AddUrl(path);
            PDB(kOutput,2) fi->Print();
            // Now add to the dataset
            dataset->Add(fi);
         }
      }
   }
   PDB(kOutput,2) Info("Merge","Done");

   // Done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Dump the class content

void TProofOutputFile::Print(Option_t *) const
{
   Info("Print","-------------- %s : start (%s) ------------", GetName(), fLocalHost.Data());
   Info("Print"," dir:              %s", fDir.Data());
   Info("Print"," raw dir:          %s", fRawDir.Data());
   Info("Print"," file name:        %s%s", fFileName.Data(), fOptionsAnchor.Data());
   if (IsMerge()) {
      Info("Print"," run type:         create a merged file");
      Info("Print"," merging option:   %s",
                       (fTypeOpt == kLocal) ? "local copy" : "keep remote");
   } else {
      TString opt;
      if ((fTypeOpt & kRegister)) opt += "R";
      if ((fTypeOpt & kOverwrite)) opt += "O";
      if ((fTypeOpt & kVerify)) opt += "V";
      Info("Print"," run type:         create dataset (name: '%s', opt: '%s')",
                                         GetTitle(), opt.Data());
   }
   Info("Print"," output file name: %s", fOutputFileName.Data());
   Info("Print"," ordinal:          %s", fWorkerOrdinal.Data());
   Info("Print","-------------- %s : done -------------", GetName());

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Notify error message

void TProofOutputFile::NotifyError(const char *msg)
{
   if (msg) {
      if (gProofServ)
         gProofServ->SendAsynMessage(msg);
      else
         Printf("%s", msg);
   } else {
      Info("NotifyError","called with empty message");
   }

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Add file to merger, checking the result

void TProofOutputFile::AddFile(TFileMerger *merger, const char *path)
{
   if (merger && path) {
      if (!merger->AddFile(path))
         NotifyError(Form("TProofOutputFile::AddFile:"
                          " error from TFileMerger::AddFile(%s)", path));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Unlink path

void TProofOutputFile::Unlink(const char *path)
{
   if (path) {
      if (!gSystem->AccessPathName(path)) {
         if (gSystem->Unlink(path) != 0)
            NotifyError(Form("TProofOutputFile::Unlink:"
                             " error from TSystem::Unlink(%s)", path));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get instance of the file collection to be used in 'dataset' mode

TFileCollection *TProofOutputFile::GetFileCollection()
{
   if (!fDataSet)
      fDataSet = new TFileCollection(GetTitle());
   return fDataSet;
}

////////////////////////////////////////////////////////////////////////////////
/// Get instance of the file merger to be used in 'merge' mode

TFileMerger *TProofOutputFile::GetFileMerger(Bool_t local)
{
   if (!fMerger)
      fMerger = new TFileMerger(local, fMergeHistosOneGo);
   return fMerger;
}

////////////////////////////////////////////////////////////////////////////////
/// Assert directory path 'dirpath', with the ownership of the last already
/// existing subpath.
/// Return 0 on success, -1 on error

Int_t TProofOutputFile::AssertDir(const char *dirpath)
{
   TString existsPath(dirpath);
   TList subPaths;
   while (existsPath != "/" && existsPath != "." && gSystem->AccessPathName(existsPath)) {
      subPaths.AddFirst(new TObjString(gSystem->BaseName(existsPath)));
      existsPath = gSystem->GetDirName(existsPath);
   }
   subPaths.SetOwner(kTRUE);
   FileStat_t st;
   if (gSystem->GetPathInfo(existsPath, st) == 0) {
      TString xpath = existsPath;
      TIter nxp(&subPaths);
      TObjString *os = 0;
      while ((os = (TObjString *) nxp())) {
         xpath += TString::Format("/%s", os->GetName());
         if (gSystem->mkdir(xpath, kTRUE) == 0) {
            if (gSystem->Chmod(xpath, (UInt_t) st.fMode) != 0)
               ::Warning("TProofOutputFile::AssertDir", "problems setting mode on '%s'", xpath.Data());
         } else {
            ::Error("TProofOutputFile::AssertDir", "problems creating path '%s'", xpath.Data());
            return -1;
         }
      }
   } else {
      ::Warning("TProofOutputFile::AssertDir", "could not get info for path '%s': will only try to create"
                           " the full path w/o trying to set the mode", existsPath.Data());
      if (gSystem->mkdir(existsPath, kTRUE) != 0) {
         ::Error("TProofOutputFile::AssertDir", "problems creating path '%s'", existsPath.Data());
         return -1;
      }
   }
   // Done
   return 0;
}
