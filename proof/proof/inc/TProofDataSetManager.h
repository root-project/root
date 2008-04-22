// @(#)root/proof:$Id$
// Author: Jan Fiete Grosse-Oetringhaus, 08.08.07

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofDataSetManager
#define ROOT_TProofDataSetManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofDataSetManager                                                 //
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
class TMD5;
class TVirtualMonitoringWriter;

class TProofDataSetManager : public TObject
{
private:

   TProofDataSetManager(const TProofDataSetManager&);             // not implemented
   TProofDataSetManager& operator=(const TProofDataSetManager&);  // not implemented

protected:

   TString fGroup;        // Group to which the owner of this session belongs
   TString fUser;         // Owner of the session
   TString fCommonUser;   // user that stores the COMMON datasets
   TString fCommonGroup;  // group that stores the COMMON datasets

   TUri    fBase;         // Base URI used to parse dataset names

   TMap    fGroupQuota;   // group quotas (read from config file)
   TMap    fGroupUsed;    // <group> --> <used bytes> (TParameter)
   TMap    fUserUsed;     // <group> --> <map of users> --> <value>

   Long64_t fAvgFileSize; // Average file size to be used to estimate the dataset size (in MB)

   Int_t   fNTouchedFiles; // number of files touched in the last ScanDataSet operation
   Int_t   fNOpenedFiles;  // number of files opened in the last ScanDataSet operation
   Int_t   fNDisappearedFiles; // number of files disappared in the last ScanDataSet operation

   TString fGroupConfigFile;  // Path to the group config file
   Long_t  fMTimeGroupConfig; // Last modification of the group config file

   static TString fgCommonDataSetTag;  // name for common datasets, default: COMMON

   virtual TMap *GetGroupUsedMap() { return &fGroupUsed; }
   virtual TMap *GetUserUsedMap() { return &fUserUsed; }
   Int_t    GetNTouchedFiles() { return fNTouchedFiles; }
   Int_t    GetNOpenedFiles() { return fNOpenedFiles; }
   Int_t    GetNDisapparedFiles() { return fNDisappearedFiles; }
   void     GetQuota(const char *group, const char *user, const char *dsName, TFileCollection *dataset);
   void     PrintUsedSpace();
   Bool_t   ReadGroupConfig(const char *cf = 0);
   static Long64_t ToBytes(const char *size = 0);
   virtual void UpdateUsedSpace();

public:
   enum EDataSetStatusBits {
      kCheckQuota    = BIT(15),   // quota checking enabled
      kAllowRegister = BIT(16),   // allow registration of a new dataset
      kAllowVerify   = BIT(17),   // allow verification of a dataset (requires registration permit)
      kAllowStaging  = BIT(18),   // allow staging of missing files (requires verification permit)
      kIsSandbox     = BIT(19)    // dataset dir is in the user sandbox (simplified naming)
   };

   enum EDataSetWorkOpts { // General (bits 1-8)
                           kDebug = 1, kShowDefault = 2, kPrint = 4, kExport = 8,
                           kQuotaUpdate = 16,
                           // File-based specific (bits 9-16)
                           kReopen = 256, kTouch = 512, kMaxFiles = 1024, kReadShort = 2048,
                           kFileMustExist = 4096};

   TProofDataSetManager(const char *group = 0, const char *user = 0, const char *options = 0);
   virtual ~TProofDataSetManager();

   static TString           CreateUri(const char *dsGroup = 0, const char *dsUser = 0,
                                      const char *dsName = 0, const char *dsTree = 0);
   virtual TFileCollection *GetDataSet(const char *uri);
   virtual TMap            *GetDataSets(const char *uri, UInt_t /*option*/ = 0);
   virtual Long64_t         GetGroupQuota(const char *group);
   virtual TMap            *GetGroupQuotaMap() { return &fGroupQuota; }
   virtual Long64_t         GetGroupUsed(const char *group);
   virtual Bool_t           ExistsDataSet(const char *uri);
   virtual void             MonitorUsedSpace(TVirtualMonitoringWriter *monitoring);
   Bool_t                   ParseUri(const char *uri, TString *dsGroup = 0, TString *dsUser = 0,
                                     TString *dsName = 0, TString *dsTree = 0,
                                     Bool_t onlyCurrent = kFALSE, Bool_t wildcards = kFALSE);
   virtual void             ParseInitOpts(const char *opts);
   virtual Bool_t           RemoveDataSet(const char *uri);
   virtual Int_t            RegisterDataSet(const char *uri, TFileCollection *dataSet, const char *opt);
   virtual Int_t            ScanDataSet(const char *uri, UInt_t /*option*/ = 0);
   virtual void             ShowQuota(const char *opt);

   ClassDef(TProofDataSetManager, 0)
};

#endif
