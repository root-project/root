// @(#)root/base:$Id$
// Author: Jan Fiete Grosse-Oetringhaus, 04.06.07

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSetManagerFile                                             //
//                                                                      //
// Implementation of TDataSetManager handling datasets from root   //
// files under a specific directory path                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDataSetManagerFile.h"

#include "Riostream.h"
#include "TEnv.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TFile.h"
#include "TFileStager.h"
#include "TLockFile.h"
#include "TMap.h"
#include "TRegexp.h"
#include "TMD5.h"
#include "TMessage.h"
#include "TSystem.h"
#include "TError.h"
#include "TPRegexp.h"
#include "TVirtualMonitoring.h"
#include "TObjArray.h"
#include "THashList.h"
#include "TKey.h"
#include "TTree.h"
#include "TParameter.h"

ClassImp(TDataSetManagerFile)

//_____________________________________________________________________________
TDataSetManagerFile::TDataSetManagerFile(const char *group,
                                         const char *user, const char *ins)
                    : TDataSetManager(group, user, ins)
{
   //
   // Main constructor

   // Parse options
   ParseInitOpts(ins);

   // Init the instance
   Init();
}

//_____________________________________________________________________________
TDataSetManagerFile::TDataSetManagerFile(const char *ins)
                    : TDataSetManager("", "", ins)
{
   //
   // Main constructor

   // Parse options
   ParseInitOpts(ins);

   // Init the instance
   Init();
}

//______________________________________________________________________________
void TDataSetManagerFile::Init()
{
   // Do the real inititialization

   if (!fUser.IsNull() && !fGroup.IsNull() && !fDataSetDir.IsNull()) {

      // Make sure that the dataset dir exists
      TString dir;
      dir.Form("%s/%s/%s", fDataSetDir.Data(), fGroup.Data(), fUser.Data());
      if (gSystem->AccessPathName(dir)) {
         if (gSystem->mkdir(dir, kTRUE) != 0) {
            TString emsg = dir;
            // Read only dataset info system: switch to COMMON
            fUser = fCommonUser;
            fGroup = fCommonGroup;
            ResetBit(TDataSetManager::kCheckQuota);
            ResetBit(TDataSetManager::kAllowRegister);
            ResetBit(TDataSetManager::kAllowVerify);
            ResetBit(TDataSetManager::kTrustInfo);
            dir.Form("%s/%s/%s", fDataSetDir.Data(), fGroup.Data(), fUser.Data());
            if (gSystem->AccessPathName(dir)) {
               Error("Init",
                     "could not attach to a valid the dataset dir; paths tried:");
               Error("Init", "    %s", emsg.Data());
               Error("Init", "    %s", dir.Data());
               SetBit(TObject::kInvalidObject);
               return;
            }
         }
      }

      // If not in sandbox, construct the base URI using session defaults
      // (group, user) (syntax: /group/user/dsname[#[subdir/]objname])
      if (!TestBit(TDataSetManager::kIsSandbox))
         fBase.SetUri(TString(Form("/%s/%s/", fGroup.Data(), fUser.Data())));

      // Fill locking path
      fDataSetLockFile = Form("%s-dataset-lock", fDataSetDir.Data());
      fDataSetLockFile.ReplaceAll("/","%");
      fDataSetLockFile.ReplaceAll(":","%");
      fDataSetLockFile.Insert(0, Form("%s/", gSystem->TempDirectory()));
   }

   // Limit in seconds after a lock automatically expires
   fLockFileTimeLimit = 120;

   // If the MSS url was not given, check if one is defined via env
   if (fMSSUrl.IsNull())
      fMSSUrl = gEnv->GetValue("ProofDataSet.MSSUrl", "");
}

//______________________________________________________________________________
void TDataSetManagerFile::ParseInitOpts(const char *ins)
{
   // Parse the input string and set the init bits accordingly
   // Format is
   //    dir:<datasetdir> [mss:<mss-url>] [opt:<base-options>]
   // The <datasetdir> is mandatory.
   // See TDataSetManager::ParseInitOpts for the available
   // base options.
   // The base options are laready initialized by the base constructor

   SetBit(TObject::kInvalidObject);

   // Needs something in
   if (!ins || strlen(ins) <= 0) return;

   // Extract elements
   Int_t from = 0;
   TString s(ins), tok;
   while (s.Tokenize(tok, from, " ")) {
      if (tok.BeginsWith("dir:"))
         fDataSetDir = tok(4, tok.Length());
      if (tok.BeginsWith("mss:"))
         fMSSUrl = tok(4, tok.Length());
   }

   // The directory is mandatory
   if (fDataSetDir.IsNull()) return;

   // Object is valid
   ResetBit(TObject::kInvalidObject);
}

//______________________________________________________________________________
const char *TDataSetManagerFile::GetDataSetPath(const char *group,
                                                     const char *user,
                                                     const char *dsName)
{
   //
   // Returns path of the indicated dataset
   // Contains a static TString for result. Copy result before using twice.

   if (fgCommonDataSetTag == group)
     group = fCommonGroup;

   if (fgCommonDataSetTag == user)
     user = fCommonUser;

   static TString result;
   result.Form("%s/%s/%s/%s.root", fDataSetDir.Data(), group, user, dsName);
   if (gDebug > 0)
      Info("GetDataSetPath","path: %s", result.Data());
   return result;
}

//______________________________________________________________________________
Bool_t TDataSetManagerFile::BrowseDataSets(const char *group,
                                                const char *user,
                                                UInt_t option, TObject *target)
{
   // Adds the dataset in the folder of group, user to the list in target
   //
   // The unsigned int 'option' is forwarded to GetDataSet and BrowseDataSet.
   // Available options (to be .or.ed):
   //    kPrint          print the dataset content
   //    kQuotaUpdate    update quotas
   //    kExport         use export naming
   //
   // NB1: options "kPrint", "kQuoatUpdate" and "kExport" are mutually exclusive
   // NB2: for options "kPrint" and "kQuotaUpdate" return is null.

   TString userDirPath;
   userDirPath.Form("%s/%s/%s", fDataSetDir.Data(), group, user);
   void *userDir = gSystem->OpenDirectory(userDirPath);
   if (!userDir)
     return kFALSE;

   // Options
   Bool_t printing = (option & kPrint) ? kTRUE : kFALSE;
   Bool_t exporting = (option & kExport) ? kTRUE : kFALSE;
   Bool_t updating = (option & kQuotaUpdate) ? kTRUE : kFALSE;

   // If printing is required add kReadShort to the options
   if (printing || updating)
      option |= kReadShort;

   // The last three options are mutually exclusive
   if (((Int_t)printing + (Int_t)exporting + (Int_t)updating) > 1) {
      Error("BrowseDataSets",
            "only one of kPrint, kQuotaUpdate or kExport can be specified at once");
      return kFALSE;
   }
   Bool_t fillmap = (!exporting && !printing && !updating) ? kTRUE : kFALSE;

   // Output object
   TMap *outmap = (fillmap || exporting) ? (TMap *)target : (TMap *)0;
   TList *outlist = (printing) ? (TList *)target : (TList *)0;

   TRegexp rg("^[^./][^/]*.root$");  //check that it is a root file, not starting with "."

   TMap *userMap = 0, *datasetMap = 0;
   // loop over datasets
   const char *dsEnt = 0;
   while ((dsEnt = gSystem->GetDirEntry(userDir))) {
      TString datasetFile(dsEnt);
      if (datasetFile.Index(rg) != kNPOS) {
         TString datasetName(datasetFile(0, datasetFile.Length()-5));

         if (gDebug > 0)
            Info("GetDataSets", "found dataset %s of user %s in group %s",
                                 datasetName.Data(), user, group);

         TFileCollection *fileList = GetDataSet(group, user, datasetName, option);
         if (!fileList) {
            Error("GetDataSets", "dataset %s (user %s, group %s) could not be opened",
                                 datasetName.Data(), user, group);
            continue;
         }
         if (gDebug > 0)
            fileList->Print();

         // We found a dataset, now add it to the map

         // COMMON dataset transition
         const char *mapGroup = group;
         if (fCommonGroup == mapGroup)
            mapGroup = fgCommonDataSetTag.Data();
         const char *mapUser = user;
         if (fCommonUser == mapUser)
            mapUser = fgCommonDataSetTag.Data();

         if (fillmap && outmap) {
            if (!(userMap = dynamic_cast<TMap*> (outmap->GetValue(mapGroup)))) {
               userMap = new TMap;
               userMap->SetOwner();
               outmap->Add(new TObjString(mapGroup), userMap);
            }

            if (!(datasetMap = dynamic_cast<TMap*> (userMap->GetValue(mapUser)))) {
               datasetMap = new TMap;
               datasetMap->SetOwner();
               userMap->Add(new TObjString(mapUser), datasetMap);
            }
         }

         // Action depends on option
         if (exporting) {

            // Just format the dataset name with group and user
            TString dsNameFormatted(Form("/%s/%s/%s", mapGroup,
                                                      mapUser, datasetName.Data()));
            if (outmap)
               outmap->Add(new TObjString(dsNameFormatted), fileList);

         } else if (updating) {

            // Update quotas
            GetQuota(mapGroup, mapUser, datasetName.Data(), fileList);

         } else if (printing) {

            // Prepare the output list
            if (outlist) {
               TString dsNameFormatted(Form("/%s/%s/%s", mapGroup,
                                                         mapUser, datasetName.Data()));
               // Magic number?
               if (dsNameFormatted.Length() < 42)
                  dsNameFormatted.Resize(42);

               // Done
               outlist->Add(fileList->ExportInfo(dsNameFormatted));
            }
         } else {
            if (fillmap && datasetMap)
               datasetMap->Add(new TObjString(datasetName), fileList);
         }
      }
   }
   gSystem->FreeDirectory(userDir);

   return kTRUE;
}

//______________________________________________________________________________
TMap *TDataSetManagerFile::GetDataSets(const char *group, const char *user,
                                            UInt_t option)
{
   // General purpose call to go through the existing datasets.
   // If <user> is 0 or "*", act on all datasets for the given <group>.
   // If <group> is 0 or "*", act on all datasets.
   // Action depends on option; available options:
   //
   //    kExport         Return a TMap object containing all the information about
   //                    datasets in the form:
   //                    { <group>, <map of users> }
   //                                     |
   //                             { <map of datasets>, <dataset>}
   //                    (<dataset> are TFileCollection objects)
   //    kShowDefault    as kExport with in addition a default selection including
   //                    the datasets from the current user, the ones from the group
   //                    and the common ones
   //
   //    kPrint          print the dataset content; no output is returned
   //    kQuotaUpdate    update {group, user} quotas; no output is returned
   //
   // NB1: options "kPrint", "kQuoatUpdate" and "kExport" are mutually exclusive
   // NB2: for options "kPrint" and "kQuoatUpdate" return is null.

   if (group && fgCommonDataSetTag == group)
     group = fCommonGroup;

   if (user && fgCommonDataSetTag == user)
     user = fCommonUser;

   // Special treatment for the COMMON user
   Bool_t notCommonUser = kTRUE;
   if ((user && fCommonUser == user) &&
       (group && fCommonGroup == group)) notCommonUser = kFALSE;

   // convert * to "nothing"
   if (group && (strcmp(group, "*") == 0 || strlen(group) == 0))
      group = 0;
   if (user && (strcmp(user, "*") == 0 || strlen(user) == 0))
      user = 0;

   Bool_t printing = (option & kPrint) ? kTRUE : kFALSE;
   Bool_t exporting = (option & kExport) ? kTRUE : kFALSE;
   Bool_t updating = (option & kQuotaUpdate) ? kTRUE : kFALSE;

   // The last three options are mutually exclusive
   if (((Int_t)printing + (Int_t)exporting + (Int_t)updating) > 1) {
      Error("GetDataSets", "only one of '?P', '?Q' or '?E' can be specified at once");
      return 0;
   }

   TObject *result = 0;
   if (printing) {
      // The output is a list of strings
      TList *ol = new TList();
      ol->SetOwner();
      result = ol;
   } else if (exporting || !updating) {
      TMap *om = new TMap;
      om->SetOwner();
      result = om;
   }

   if (gDebug > 0)
      Info("GetDataSets", "opening dir %s", fDataSetDir.Data());

   if (option & kShowDefault) {
      // add the common ones
      BrowseDataSets(fCommonGroup, fCommonUser, option, result);
      user = 0;
   } else {
      // Fill the information at least once
      if (!notCommonUser) notCommonUser = kTRUE;
   }

   // Fill the information only once
   if (notCommonUser) {
      // group, user defined, no looping needed
      if (user && group) {
         BrowseDataSets(group, user, option, result);
         if (!printing) return (TMap *)result;
      } else {
         // loop needed
         void *dataSetDir = 0;
         if ((dataSetDir = gSystem->OpenDirectory(fDataSetDir))) {
            // loop over groups
            const char *currentGroup = 0;
            while ((currentGroup = gSystem->GetDirEntry(dataSetDir))) {

               if (strcmp(currentGroup, ".") == 0 || strcmp(currentGroup, "..") == 0)
                  continue;

               if (group && strcmp(group, currentGroup))
                  continue;

               TString groupDirPath;
               groupDirPath.Form("%s/%s", fDataSetDir.Data(), currentGroup);

               void *groupDir = gSystem->OpenDirectory(groupDirPath);
               if (!groupDir)
                  continue;

               // loop over users
               const char *currentUser = 0;
               while ((currentUser = gSystem->GetDirEntry(groupDir))) {

                  if (strcmp(currentUser, ".") == 0 || strcmp(currentUser, "..") == 0)
                     continue;

                  if (user && strcmp(user, currentUser))
                     continue;

                  BrowseDataSets(currentGroup, currentUser, option, result);
               }
               gSystem->FreeDirectory(groupDir);
            }
            gSystem->FreeDirectory(dataSetDir);
         }
      }
   }
   // Print the result, if required
   if (printing) {
      TList *output = (TList *)result;
      output->Sort();
      Printf("Dataset repository: %s", fDataSetDir.Data());
      Printf("Dataset URI                               | # Files | Default tree | # Events |   Disk   | Staged");
      TIter iter4(output);
      TObjString* formattedLine = 0;
      while ((formattedLine = dynamic_cast<TObjString*> (iter4())))
         Printf("%s", formattedLine->String().Data());
      // Cleanup
      SafeDelete(output);
      result = 0;
   }

   return (TMap *)result;
}

//______________________________________________________________________________
TFileCollection *TDataSetManagerFile::GetDataSet(const char *group,
                                                      const char *user,
                                                      const char *dsName,
                                                      UInt_t option,
                                                      TMD5 **checksum)
{
   //
   // Returns the dataset <dsName> of user <user> in group <group> .
   // If checksum is non-zero, it will contain the pointer to a TMD5 sum object
   // with the checksum of the file, has to be deleted by the user.
   // If option has the bi kReadShort set, the shortobject is read, that does not
   // contain the list of files. This is much faster.

   TLockFile lock(fDataSetLockFile, fLockFileTimeLimit);

   TString path(GetDataSetPath(group, user, dsName));

   if (gSystem->AccessPathName(path) != kFALSE) {
      Info("GetDataSet", "dataset %s does not exist", path.Data());
      return 0;
   }

   TMD5 *retrievedChecksum = 0;
   if (checksum) {
      // save md5 sum
      retrievedChecksum = TMD5::FileChecksum(path);
      if (!retrievedChecksum) {
          Error("GetDataSet", "could not get checksum of %s", path.Data());
         return 0;
      }
   }

   TFile *f = TFile::Open(path.Data());
   if (!f) {
      Error("GetDataSet", "Could not open file %s", path.Data());
      if (retrievedChecksum)
         delete retrievedChecksum;
      return 0;
   }

   TFileCollection *fileList = 0;
   if (option & kReadShort)
     fileList = dynamic_cast<TFileCollection*> (f->Get("dataset_short"));

   if (!fileList)
     fileList = dynamic_cast<TFileCollection*> (f->Get("dataset"));

   f->Close();
   delete f;

   if (checksum)
      *checksum = retrievedChecksum;
   return fileList;
}

//______________________________________________________________________________
Int_t TDataSetManagerFile::WriteDataSet(const char *group, const char *user,
                                         const char *dsName, TFileCollection *dataset,
                                         UInt_t option, TMD5 *checksum)
{
   //
   // Writes indicated dataset.
   // If option has the bit kFileMustExist set, the file must still exist,
   // otherwise the new dataset is not written (returns 3 in this case).
   // If checksum is non-zero the files current checksum is checked against it,
   // if it does not match the file is not written (the function returns 2 in this
   // case, if the file has disappeared it is also not written (i.e. checksum
   // implies the bit kFileMustExist set in option).
   // Returns != 0 for success, 0 for error

   TLockFile lock(fDataSetLockFile, fLockFileTimeLimit);

   Bool_t checkIfExists = ((option & kFileMustExist) || checksum) ? kTRUE : kFALSE;

   TString path(GetDataSetPath(group, user, dsName));

   if (checkIfExists) {
      // check if file still exists, otherwise it was deleted in the meanwhile and is not written here
      Long_t tmp;
      if (gSystem->GetPathInfo(path, 0, (Long_t*) 0, 0, &tmp) != 0) {
         if (gDebug > 0)
            Info("WriteDataSet", "Dataset disappeared. Discarding update.");
         return 3;
      }
   }

   if (checksum) {
      // verify md5 sum, otherwise the file was changed in the meanwhile and is not overwritten here
      TMD5 *checksum2 = TMD5::FileChecksum(path);
      if (!checksum2) {
         Error("WriteDataSet", "Could not get checksum of %s", path.Data());
         return 0;
      }

      Bool_t checksumAgrees = (*checksum == *checksum2);
      delete checksum2;

      if (!checksumAgrees) {
         if (gDebug > 0)
            Info("WriteDataSet", "Dataset changed. Discarding update.");
         return 2;
      }
   }

   // write first in ".<file>" then rename to recover from crash during writing
   TString tempFile(path);
   Int_t index = -1;
   while (tempFile.Index("/", index+1) >= 0)
      index = tempFile.Index("/", index+1);

   tempFile.Insert(index+1, ".");

   TFile *f = TFile::Open(tempFile, "RECREATE");
   if (!f) {
      Error("WriteDataSet", "Could not open dataset for writing %s", tempFile.Data());
      return 0;
   }

   // write full TFileCollection
   dataset->Write("dataset", TObject::kSingleKey);

   // write only metadata
   THashList *list = dataset->GetList();
   dataset->SetList(0);
   dataset->Write("dataset_short", TObject::kSingleKey);

   f->Close();
   delete f;

   dataset->SetList(list);

   // file is written, rename to real filename
   if (gSystem->Rename(tempFile, path) != 0) {
      Error("WriteDataSet", "Renaming %s to %s failed. Dataset might be corrupted.",
                            tempFile.Data(), path.Data());
      return 0;
   }

   return 1;
}

//______________________________________________________________________________
Bool_t TDataSetManagerFile::RemoveDataSet(const char *group, const char *user,
                                               const char *dsName)
{
   // Removes the indicated dataset

   TLockFile lock(fDataSetLockFile, fLockFileTimeLimit);

   TString path(GetDataSetPath(group, user, dsName));

   return (gSystem->Unlink(path) == 0);
}

//______________________________________________________________________________
Bool_t TDataSetManagerFile::ExistsDataSet(const char *group, const char *user,
                                               const char *dsName)
{
   // Checks if the indicated dataset exits

   TLockFile lock(fDataSetLockFile, fLockFileTimeLimit);

   TString path(GetDataSetPath(group, user, dsName));

   return (gSystem->AccessPathName(path) == kFALSE);
}

//______________________________________________________________________________
Int_t TDataSetManagerFile::RegisterDataSet(const char *uri,
                                           TFileCollection *newDataSet,
                                           const char *opts)
{
   // Register a dataset, perfoming quota checkings and verification, if required.
   // If a dataset with the same name already exists the action fails unless 'opts'
   // contains 'O', in which case the old dataset is overwritten, or contains 'U',
   // in which case 'newDataSet' is added to the existing dataset (duplications are
   // ignored, if any).
   // If 'opts' contains 'V' the dataset files are also verified (if the dataset manager
   // is configured to allow so). By default the dataset is not verified.
   // If 'opts' contains 'T' the in the dataset object (status bits, meta,...)
   // is trusted, i.e. not reset (if the dataset manager is configured to allow so).
   // Returns 0 on success, -1 on failure

   if (!TestBit(TDataSetManager::kAllowRegister))
      return -1;

   // Get the dataset name
   TString dsName;
   if (ParseUri(uri, 0, 0, &dsName, 0, kTRUE) == kFALSE) {
      Error("RegisterDataSet", "problem parsing uri: %s", uri);
      return -1;
   }

   // The dataset
   TFileCollection *dataSet = newDataSet;
   // Check option
   TString opt(opts);
   // If in update mode, retrieve the existing dataset, if any
   if (opt.Contains("U", TString::kIgnoreCase)) {
      // Fail if it exists already
      if (ExistsDataSet(fGroup, fUser, dsName)) {
         // Retrieve the dataset
         if (!(dataSet = GetDataSet(fGroup, fUser, dsName))) {
            // Dataset name does exist
            Warning("RegisterDataSet",
                    "dataset '%s' claimed to exists but retrieval failed - ignoring", uri);
            dataSet = newDataSet;
         } else {
            // Add new dataset to existing one
            dataSet->Add(newDataSet);
         }
      }
   } else if (!opt.Contains("O", TString::kIgnoreCase)) {
      // Fail if it exists already
      if (ExistsDataSet(fGroup, fUser, dsName)) {
         //Dataset name does exist
         Error("RegisterDataSet", "dataset '%s' exists already", uri);
         return -1;
      }
   }

   // We will save a sorted list
   dataSet->Sort();

   // A temporary list to hold the unique members (i.e. the very set)
   TList *uniqueFileList = new TList();
   TIter nextFile(dataSet->GetList());
   TFileInfo *prevFile = (TFileInfo*)nextFile();
   uniqueFileList->Add(prevFile);
   while (TFileInfo *obj = (TFileInfo*)nextFile()) {
      // Add entities only once to the temporary list
      if (prevFile->Compare(obj)) {
         uniqueFileList->Add(obj);
         prevFile = obj;
      }
   }

   // Clear dataSet and add contents of uniqueFileList needed otherwise
   // THashList deletes the objects even when nodelete is set
   dataSet->GetList()->SetOwner(0);
   dataSet->GetList()->Clear("nodelete");
   dataSet->GetList()->SetOwner(1);
   dataSet->GetList()->AddAll(uniqueFileList);
   uniqueFileList->SetOwner(kFALSE);
   delete uniqueFileList;

   // Enforce certain settings
   Bool_t reset = kTRUE;
   if (opt.Contains("T", TString::kIgnoreCase)) {
      if (!TestBit(TDataSetManager::kTrustInfo)) {
         Warning("RegisterDataSet", "configured to not trust the information"
                                    " provided by users: ignoring request");
      } else {
         reset = kFALSE;
      }
   }
   if (reset) {
      dataSet->SetName(dsName);
      dataSet->ResetBitAll(TFileInfo::kStaged);
      dataSet->ResetBitAll(TFileInfo::kCorrupted);
      dataSet->RemoveMetaData();
   }

   // Verify the dataset if required
   if (opt.Contains("V", TString::kIgnoreCase)) {
      if (TestBit(TDataSetManager::kAllowVerify)) {
         // Reopen files and notify
         if (TDataSetManager::ScanDataSet(dataSet, 1, kTRUE ) < 0) {
            Error("RegisterDataSet", "problems verifying the dataset");
            return -1;
         }
      } else {
         Warning("RegisterDataSet", "user-driven verification not allowed: ignoring request");
      }
   }

   // Update accumulated information
   dataSet->Update(fAvgFileSize);

   if (TestBit(TDataSetManager::kCheckQuota)) {
      if (dataSet->GetTotalSize() <= 0) {
         Error("RegisterDataSet", "datasets without size information are not accepted:");
         if (fAvgFileSize < 0) {
            Error("RegisterDataSet", "you may want to define an average"
                                     " file size to get an estimated dataset size");
         }
         return -1;
      }
      // now check the quota
      UpdateUsedSpace();
      Long64_t used = GetGroupUsed(fGroup) + dataSet->GetTotalSize();

      Info("RegisterDataSet", "your group %s uses %.1f GB + %.1f GB for the new dataset; "
                              "the available quota is %.1f GB", fGroup.Data(),
                              (Float_t) GetGroupUsed(fGroup)    / 1073741824,
                              (Float_t) dataSet->GetTotalSize() / 1073741824,
                              (Float_t) GetGroupQuota(fGroup)   / 1073741824);
      if (used > GetGroupQuota(fGroup)) {
         Error("RegisterDataSet", "quota exceeded");
         return -1;
      }
   }

   Bool_t success = WriteDataSet(fGroup, fUser, dsName, dataSet);
   if (!success)
      Error("RegisterDataSet", "could not write dataset: %s", dsName.Data());

   // Done
   return ((success) ? 0 : -1);
}
//______________________________________________________________________________
Int_t TDataSetManagerFile::ScanDataSet(const char *uri, UInt_t opt)
{
   // Scans the dataset indicated by <uri> and returns the number of missing files.
   // Returns -1 if any failure occurs, >= 0 on success.
   // For more details, see documentation of
   // ScanDataSet(TFileCollection *dataset, const char *option)

   TString dsName, dsTree;
   if ((opt & kSetDefaultTree)) {
      if (TestBit(TDataSetManager::kAllowRegister)) {
         if (ParseUri(uri, 0, 0, &dsName, &dsTree, kTRUE)) {
            TFileCollection *dataset = GetDataSet(fGroup, fUser, dsName);
            if (!dataset) return -1;
            dataset->SetDefaultTreeName(dsTree.Data());
            Int_t rc = WriteDataSet(fGroup, fUser, dsName, dataset);
            delete dataset;
            return (rc == 0) ? -1 : 0;
         }
      }
   } else {
      if (TestBit(TDataSetManager::kAllowVerify)) {
         if (ParseUri(uri, 0, 0, &dsName, 0, kTRUE)) {
            if (ScanDataSet(fGroup, fUser, dsName, (UInt_t)(kReopen | kDebug)) > 0)
               return GetNDisapparedFiles();
         }
      }
   }
   return -1;
}

//______________________________________________________________________________
Int_t TDataSetManagerFile::ScanDataSet(const char *group, const char *user,
                                       const char *dsName, UInt_t option)
{
   // See documentation of ScanDataSet(TFileCollection *dataset, UInt_t option)

   if (!TestBit(TDataSetManager::kAllowVerify))
      return -1;

   TFileCollection *dataset = GetDataSet(group, user, dsName);
   if (!dataset)
      return -1;

   // Scan controllers
   Int_t fopenopt = 0;
   if ((option & kReopen)) fopenopt++;
   if ((option & kTouch)) fopenopt++;
   Bool_t notify = ((option & kDebug)) ? kTRUE : kFALSE;
   // Do the scan
   Int_t result = TDataSetManager::ScanDataSet(dataset, fopenopt,
                                   notify, 0, (TList *)0, fAvgFileSize, fMSSUrl.Data(), -1,
                                   &fNTouchedFiles, &fNOpenedFiles, &fNDisappearedFiles);
   if (result == 2) {
      if (WriteDataSet(group, user, dsName, dataset) == 0) {
         delete dataset;
         return -2;
      }
   }
   delete dataset;

   return result;
}

//______________________________________________________________________________
TMap *TDataSetManagerFile::GetDataSets(const char *uri, UInt_t option)
{
   //
   // Returns all datasets for the <group> and <user> specified by <uri>.
   // If <user> is 0, it returns all datasets for the given <group>.
   // If <group> is 0, it returns all datasets.
   // The returned TMap contains:
   //    <group> --> <map of users> --> <map of datasets> --> <dataset> (TFileCollection)
   //
   // The unsigned int 'option' is forwarded to GetDataSet and BrowseDataSet.
   // Available options (to be .or.ed):
   //    kShowDefault    a default selection is shown that include the ones from
   //                    the current user, the ones from the group and the common ones
   //    kPrint          print the dataset content
   //    kQuotaUpdate    update quotas
   //    kExport         use export naming
   //
   // NB1: options "kPrint", "kQuoatUpdate" and "kExport" are mutually exclusive
   // NB2: for options "kPrint" and "kQuoatUpdate" return is null.

   TString dsUser, dsGroup;

   if (((option & kPrint) || (option & kExport)) && strlen(uri) <= 0)
      option |= kShowDefault;

   if (ParseUri(uri, &dsGroup, &dsUser, 0, 0, kFALSE, kTRUE))
      return GetDataSets(dsGroup, dsUser, option);
   return (TMap *)0;
}

//______________________________________________________________________________
TFileCollection *TDataSetManagerFile::GetDataSet(const char *uri, const char *srv)
{
   // Utility function used in various methods for user dataset upload.

   TString dsUser, dsGroup, dsName;

   if (!ParseUri(uri, &dsGroup, &dsUser, &dsName))
      return (TFileCollection *)0;
   TFileCollection *fc = GetDataSet(dsGroup, dsUser, dsName);

   if (fc && srv && strlen(srv) > 0) {
      // Build up the subset
      TFileCollection *sfc = 0;
      TString ss(srv), s;
      Int_t from = 0;
      while (ss.Tokenize(s, from, ",")) {
         TFileCollection *xfc = fc->GetFilesOnServer(s.Data());
         if (xfc) {
            if (sfc) {
               sfc->Add(xfc);
               delete xfc;
            } else {
               sfc = xfc;
            }
         }
      }
      // Cleanup
      delete fc;
      fc = sfc;
   }
   // Done
   return fc;
}

//______________________________________________________________________________
Bool_t TDataSetManagerFile::RemoveDataSet(const char *uri)
{
   // Removes the indicated dataset

   TString dsName;

   if (TestBit(TDataSetManager::kAllowRegister)) {
      if (ParseUri(uri, 0, 0, &dsName, 0, kTRUE)) {
         Bool_t rc = RemoveDataSet(fGroup, fUser, dsName);
         if (rc) return kTRUE;
         Error("RemoveDataSet", "error removing dataset %s", dsName.Data());
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TDataSetManagerFile::ExistsDataSet(const char *uri)
{
   // Checks if the indicated dataset exits

   TString dsUser, dsGroup, dsName;

   if (ParseUri(uri, &dsGroup, &dsUser, &dsName))
      return ExistsDataSet(dsGroup, dsUser, dsName);
   return kFALSE;
}

//______________________________________________________________________________
void TDataSetManagerFile::UpdateUsedSpace()
{
   // updates the used space maps

   // Clear used space entries
   fGroupUsed.DeleteAll();
   fUserUsed.DeleteAll();

   // Scan the existing datasets
   GetDataSets(0, 0, (UInt_t)kQuotaUpdate);
}
