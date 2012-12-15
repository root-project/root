// @(#)root/proof:$Id$
// Author: Jan Fiete Grosse-Oetringhaus, 08.08.07

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataSetManagerFile
#define ROOT_TDataSetManagerFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSetManagerFile                                                  //
//                                                                      //
// Implementation of TDataSetManager handling datasets from root        //
// files under a specific directory path                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDataSetManager
#include "TDataSetManager.h"
#endif

const char* const kDataSet_LocalCache   = "dataset.cache"; // default cache subdirectory
const char* const kDataSet_DataSetList  = "dataset.list";  // file with info about all datasets
const char* const kDataSet_LockLocation = "lock.location"; // location of the lock file

class TDataSetManagerFile : public TDataSetManager {

private:
   TString fDataSetDir;        // Location of datasets
   TString fMSSUrl;            // URL for the Mass Storage System
   TString fStageOpts;         // Option string to be used in issuing staging requests
   TString fDataSetLockFile;   // Dataset lock file
   Int_t   fLockFileTimeLimit; // Limit in seconds after a lock automatically expires
   TString fListFile;          // File to check repository updates
   Bool_t  fIsRemote;          // True if the repository is remote
   Bool_t  fUseCache;          // True if the cache is used for browsing remote repositories
   TString fLocalCacheDir;     // Local cache when the repository is remote
   Int_t   fCacheUpdatePeriod; // Period for checking for new updated information
   Bool_t  fOpenPerms;         // True if file permissions must be open

   // Local cache handling
   void    InitLocalCache();
   Int_t   CheckLocalCache(const char *group, const char *user, const char *dsName = "ls", UInt_t option = 0);

protected:
   const char *GetDataSetPath(const char *group, const char *user, const char *dsName);
   const char *GetDataSetPath(const char *group, const char *user, const char *dsName,
                              TString &md5path, Bool_t local = kFALSE);
   void   Init();
   Bool_t BrowseDataSets(const char *group, const char *user, const char *dsName,
                         UInt_t option, TObject *target);

   Bool_t RemoveDataSet(const char *group, const char *user, const char *dsName);
   Bool_t ExistsDataSet(const char *group, const char *user, const char *dsName);

   Int_t  ScanDataSet(const char *group, const char *user, const char *dsName, UInt_t option = kReopen | kDebug);

   Int_t  ChecksumDataSet(const char *path, const char *md5path, TString &checksum);

   Int_t  CreateLsFile(const char *group, const char *user, Long_t &mtime, TString &checksum);
   Int_t  FillLsDataSet(const char *group, const char *user, const char *dsName, TList *out, UInt_t option);

   void UpdateUsedSpace();

public:
   TDataSetManagerFile() : TDataSetManager(0, 0, 0) { }
   TDataSetManagerFile(const char *group, const char *user, const char *ins);
   TDataSetManagerFile(const char *ins);
   virtual ~TDataSetManagerFile() { }

   void             ParseInitOpts(const char *opts);

   Int_t            ClearCache(const char *uri = 0);
   TFileCollection *GetDataSet(const char *uri, const char *srv = 0);
   TMap            *GetDataSets(const char *uri, UInt_t option = TDataSetManager::kExport);
   Bool_t           ExistsDataSet(const char *uri);
   Bool_t           RemoveDataSet(const char *uri);

   Int_t            RegisterDataSet(const char *uri, TFileCollection *dataSet, const char *opt);
   Int_t            ScanDataSet(const char *uri, UInt_t option = kReopen | kDebug);
   Int_t            NotifyUpdate(const char *group, const char *user,
                                 const char *dspath, Long_t mtime, const char *checksum = 0);
   Int_t            ShowCache(const char *uri = 0);

   // These should / could be private but they are used directly by the external daemon
   TFileCollection *GetDataSet(const char *group, const char *user, const char *dsName,
                               UInt_t option = 0, TMD5 **checksum = 0);
   TMap            *GetDataSets(const char *group, const char *user, const char *dsName = 0,
                                UInt_t option = 0);
   const char      *GetMSSUrl() const { return fMSSUrl; }
   const char      *GetStageOpts() const { return fStageOpts; }
   Int_t            WriteDataSet(const char *group, const char *user, const char *dsName,
                                 TFileCollection *dataset, UInt_t option = 0, TMD5 *checksum = 0);
   Long_t           GetModTime(const char *uri);

   ClassDef(TDataSetManagerFile, 0) // DataSet manager for files
};

#endif
