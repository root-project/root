// @(#)root/proof:$Id$
// Author: Jan Fiete Grosse-Oetringhaus, 08.08.07

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataSetManager
#define ROOT_TDataSetManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSetManager                                                 //
//                                                                      //
// This class contains functions to handle datasets in PROOF            //
// It is the layer between TProofServ and the file system that stores   //
// the datasets.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"
#include "TMap.h"
#include "TUri.h"


class TFileCollection;
class TFileInfo;
class TMD5;
class TUrl;
class TVirtualMonitoringWriter;
class TFileStager;

class TDataSetManager : public TObject {

private:
   TDataSetManager(const TDataSetManager&);             // not implemented
   TDataSetManager& operator=(const TDataSetManager&);  // not implemented

protected:
   TString  fGroup;         // Group to which the owner of this session belongs
   TString  fUser;          // Owner of the session
   TString  fCommonUser;    // User that stores the COMMON datasets
   TString  fCommonGroup;   // Group that stores the COMMON datasets

   TUri     fBase;          // Base URI used to parse dataset names

   TMap     fGroupQuota;    // Group quotas (read from config file)
   TMap     fGroupUsed;     // <group> --> <used bytes> (TParameter)
   TMap     fUserUsed;      // <group> --> <map of users> --> <value>

   Long64_t fAvgFileSize;   // Average file size to be used to estimate the dataset size (in MB)

   Int_t    fNTouchedFiles; // Number of files touched in the last ScanDataSet operation
   Int_t    fNOpenedFiles;  // Number of files opened in the last ScanDataSet operation
   Int_t    fNDisappearedFiles; // Number of files disappared in the last ScanDataSet operation

   TString  fGroupConfigFile;  // Path to the group config file
   Long_t   fMTimeGroupConfig; // Last modification of the group config file

   static TString fgCommonDataSetTag;  // Name for common datasets, default: COMMON

   static TList *fgDataSetSrvMaps; // List of TPair(TRegexp, TObjString) for mapping server coordinates
                                  // for dataset files (init from DataSet.SrvMap)

   virtual TMap *GetGroupUsedMap() { return &fGroupUsed; }
   virtual TMap *GetUserUsedMap() { return &fUserUsed; }
   Int_t    GetNTouchedFiles() const { return fNTouchedFiles; }
   Int_t    GetNOpenedFiles() const { return fNOpenedFiles; }
   Int_t    GetNDisapparedFiles() const { return fNDisappearedFiles; }
   void     GetQuota(const char *group, const char *user, const char *dsName, TFileCollection *dataset);
   void     PrintDataSet(TFileCollection *fc, Int_t popt = 0);
   void     PrintUsedSpace();
   Bool_t   ReadGroupConfig(const char *cf = 0);
   virtual void UpdateUsedSpace();

   static Long64_t ToBytes(const char *size = 0);

public:
   enum EDataSetStatusBits {
      kCheckQuota    = BIT(15),   // quota checking enabled
      kAllowRegister = BIT(16),   // allow registration of a new dataset
      kAllowVerify   = BIT(17),   // allow verification of a dataset (requires registration permit)
      kTrustInfo     = BIT(18),   // during registration, trust the available information provided by the user
      kIsSandbox     = BIT(19),   // dataset dir is in the user sandbox (simplified naming)
      kUseCache      = BIT(20),   // force the usage of cache
      kDoNotUseCache = BIT(21)    // disable the cache
   };

   enum EDataSetWorkOpts { // General (bits 1-8)
                           kDebug = 0x1, kShowDefault = 0x2, kPrint = 0x4, kExport = 0x8,
                           kQuotaUpdate = 0x10, kSetDefaultTree = 0x20, kForceScan = 0x40,
                           kNoHeaderPrint = 0x80,
                           // File-based specific (bits 9-16)
                           kReopen = 0x100, kTouch = 0x200, kMaxFiles = 0x400, kReadShort = 0x800,
                           kFileMustExist = 0x1000,
                           kNoAction = 0x2000, kLocateOnly = 0x4000, kStageOnly = 0x8000,
                           // Auxilliary bits (bits 17-)
                           kNoCacheUpdate = 0x10000, kRefreshLs = 0x20000, kList = 0x40000,
                           kAllFiles = 0x80000, kStagedFiles = 0x100000, kNoStagedCheck = 0x200000
                           };

   TDataSetManager(const char *group = 0, const char *user = 0, const char *options = 0);
   virtual ~TDataSetManager();

   virtual Int_t            ClearCache(const char *uri);
   virtual Long64_t         GetAvgFileSize() const { return fAvgFileSize; }
   virtual TFileCollection *GetDataSet(const char *uri, const char *server = 0);
   virtual TMap            *GetDataSets(const char *uri, UInt_t /*option*/ = TDataSetManager::kExport);
   virtual TMap            *GetSubDataSets(const char *uri, const char *excludeservers);

   virtual Long64_t         GetGroupQuota(const char *group);
   virtual TMap            *GetGroupQuotaMap() { return &fGroupQuota; }
   virtual Long64_t         GetGroupUsed(const char *group);
   virtual Bool_t           ExistsDataSet(const char *uri);
   virtual void             MonitorUsedSpace(TVirtualMonitoringWriter *monitoring);
   virtual Int_t            NotifyUpdate(const char *group = 0, const char *user = 0,
                                         const char *dspath = 0, Long_t mtime = 0, const char *checksum = 0);
   Bool_t                   ParseUri(const char *uri, TString *dsGroup = 0, TString *dsUser = 0,
                                     TString *dsName = 0, TString *dsTree = 0,
                                     Bool_t onlyCurrent = kFALSE, Bool_t wildcards = kFALSE);
   virtual void             ParseInitOpts(const char *opts);
   virtual Bool_t           RemoveDataSet(const char *uri);
   virtual Int_t            RegisterDataSet(const char *uri, TFileCollection *dataSet, const char *opt);
   Int_t                    ScanDataSet(const char *uri, const char *opt);
   virtual Int_t            ScanDataSet(const char *uri, UInt_t option = kReopen | kDebug);
   void                     SetScanCounters(Int_t t = -1, Int_t o = -1, Int_t d = -1);
   virtual Int_t            ShowCache(const char *uri);
   virtual void             ShowQuota(const char *opt);

   virtual void             ShowDataSets(const char *uri = "*", const char *opt = "");

   static TString           CreateUri(const char *dsGroup = 0, const char *dsUser = 0,
                                      const char *dsName = 0, const char *dsTree = 0);
   static Bool_t            CheckDataSetSrvMaps(TUrl *furl, TString &fn, TList *srvmaplist = 0);
   static TList            *GetDataSetSrvMaps();
   static TList            *ParseDataSetSrvMaps(const TString &srvmaps);
   static Int_t             ScanDataSet(TFileCollection *dataset, Int_t fopt, Int_t sopt = 0, Int_t ropt = 0,
                                        Bool_t dbg = kFALSE,
                                        Int_t *touched = 0, Int_t *opened = 0, Int_t *disappeared = 0,
                                        TList *flist = 0, Long64_t avgsz = -1, const char *mss = 0,
                                        Int_t maxfiles = -1, const char *stageopts = 0);
   static Bool_t            CheckStagedStatus(TFileInfo *fileInfo, Int_t fopt, Int_t maxfiles,
                                              Int_t newstagedfiles, TFileStager* stager, Bool_t createStager,
                                              Bool_t dbg, Bool_t &changed, Bool_t& touched,
                                              Bool_t& disappeared);
   static void              ProcessFile(TFileInfo *fileInfo, Int_t sopt, Bool_t checkstg, Bool_t doall,
                                        TFileStager* stager, Bool_t createStager, const char* stageopts,
                                        Bool_t dbg, Bool_t &changed, Bool_t& opened);

   static Int_t             ScanFile(TFileInfo *fileinfo, Bool_t notify);
   static Int_t             FillMetaData(TFileInfo *fi, TDirectory *d, const char *rdir = "/");

   ClassDef(TDataSetManager, 0)  // Abstract data set manager class
};

#endif
