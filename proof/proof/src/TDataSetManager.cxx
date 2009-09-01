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
// TDataSetManager                                                 //
//                                                                      //
// This class contains functions to handle datasets in PROOF            //
// It is the layer between TProofServ and the file system that stores   //
// the datasets.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TDataSetManager.h"

#include "Riostream.h"

#include "TEnv.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TMD5.h"
#include "THashList.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TPRegexp.h"
#include "TRegexp.h"
#include "TSystem.h"
#include "TVirtualMonitoring.h"

// One Gigabyte
#define DSM_ONE_GB (1073741824)

// Name for common datasets
TString TDataSetManager::fgCommonDataSetTag = "COMMON";

ClassImp(TDataSetManager)

//_____________________________________________________________________________
TDataSetManager::TDataSetManager(const char *group, const char *user,
                                           const char *options)
                     : fGroup(group),
                       fUser(user), fCommonUser(), fCommonGroup(),
                       fGroupQuota(), fGroupUsed(),
                       fUserUsed(), fNTouchedFiles(0), fNOpenedFiles(0),
                       fNDisappearedFiles(0), fMTimeGroupConfig(-1)
{
   //
   // Main constructor

   // Fill default group and user if none is given
   if (fGroup.IsNull())
      fGroup = "default";
   if (fUser.IsNull()) {
      fUser = "--nouser--";
      // Get user logon name
      UserGroup_t *pw = gSystem->GetUserInfo();
      if (pw) {
         fUser = pw->fUser;
         delete pw;
      }
   }

   fGroupQuota.SetOwner();
   fGroupUsed.SetOwner();
   fUserUsed.SetOwner();

   fCommonUser = "COMMON";
   fCommonGroup = "COMMON";

   fNTouchedFiles = -1;
   fNOpenedFiles = -1;
   fNDisappearedFiles = -1;
   fMTimeGroupConfig = -1;

   fAvgFileSize = 50000000;  // Default 50 MB per file

   // Parse options
   ParseInitOpts(options);

   if (!fUser.IsNull() && !fGroup.IsNull()) {

      // If not in sandbox, construct the base URI using session defaults
      // (group, user) (syntax: /group/user/dsname[#[subdir/]objname])
      if (!TestBit(TDataSetManager::kIsSandbox))
         fBase.SetUri(TString(Form("/%s/%s/", fGroup.Data(), fUser.Data())));

   }

   // Read config file
   ReadGroupConfig(gEnv->GetValue("Proof.GroupFile", ""));
}

//______________________________________________________________________________
TDataSetManager::~TDataSetManager()
{
   // Destructor

   // Clear used space
   fGroupQuota.DeleteAll();
   fGroupUsed.DeleteAll();
   fUserUsed.DeleteAll();
}

//______________________________________________________________________________
void TDataSetManager::ParseInitOpts(const char *opts)
{
   // Parse the opts string and set the init bits accordingly
   // Available options:
   //    Cq:               set kCheckQuota
   //    Ar:               set kAllowRegister
   //    Av:               set kAllowVerify
   //    As:               set kAllowStaging
   //    Sb:               set kIsSandbox
   // The opts string may also contain additional unrelated info: in such a case
   // the field delimited by the prefix "opt:" is analyzed, e.g. if opts is
   // "/tmp/dataset  opt:Cq:-Ar: root://lxb6046.cern.ch" only the substring
   // "Cq:-Ar:" will be parsed .

   // Default option bits
   ResetBit(TDataSetManager::kCheckQuota);
   SetBit(TDataSetManager::kAllowRegister);
   SetBit(TDataSetManager::kAllowVerify);
   ResetBit(TDataSetManager::kAllowStaging);
   ResetBit(TDataSetManager::kIsSandbox);

   if (opts && strlen(opts) > 0) {
      TString opt(opts);
      // If it contains the prefix "opt:", isolate the related field
      Int_t ip = opt.Index("opt:");
      if (ip != kNPOS) opt.Remove(0, ip + 4);
      ip = opt.Index(" ");
      if (ip != kNPOS) opt.Remove(ip);
      // Check the content, now
      if (opt.Contains("Cq:") && !opt.Contains("-Cq:"))
         SetBit(TDataSetManager::kCheckQuota);
      if (opt.Contains("-Ar:"))
         ResetBit(TDataSetManager::kAllowRegister);
      if (opt.Contains("-Av:"))
         ResetBit(TDataSetManager::kAllowVerify);
      if (opt.Contains("As:") && !opt.Contains("-As:"))
         SetBit(TDataSetManager::kAllowStaging);
      if (opt.Contains("Sb:") && !opt.Contains("-Sb:"))
         SetBit(TDataSetManager::kIsSandbox);
   }

   // Check dependencies
   if (TestBit(TDataSetManager::kAllowStaging)) {
      // Staging of missing files requires verification permition
      SetBit(TDataSetManager::kAllowVerify);
   }
   if (TestBit(TDataSetManager::kAllowVerify)) {
      // Dataset verification requires registration permition
      SetBit(TDataSetManager::kAllowRegister);
   }
}

//______________________________________________________________________________
Bool_t TDataSetManager::ReadGroupConfig(const char *cf)
{
   // Read group config file 'cf'.
   // If cf == 0 re-read, if changed, the file pointed by fGroupConfigFile .
   //
   // expects the following directives:
   // Group definition:
   //   group <groupname> <user>+
   // disk quota
   //   property <groupname> diskquota <quota in GB>
   // average filesize (to be used when the file size is not available)
   //   averagefilesize <average size>{G,g,M,m,K,k}

   // Validate input
   FileStat_t st;
   if (!cf || (strlen(cf) <= 0) || !strcmp(cf, fGroupConfigFile.Data())) {
      // If this is the first time we cannot do anything
      if (fGroupConfigFile.IsNull()) {
         if (gDebug > 0)
            Info("ReadGroupConfig", "path to config file undefined - nothing to do");
         return kFALSE;
      }
      // Check if fGroupConfigFile has changed
      if (gSystem->GetPathInfo(fGroupConfigFile, st)) {
         Error("ReadGroupConfig", "could not stat %s", fGroupConfigFile.Data());
         return kFALSE;
      }
      if (st.fMtime <= fMTimeGroupConfig) {
         if (gDebug > 0)
            Info("ReadGroupConfig","file has not changed - do nothing");
         return kTRUE;
      }
   }

   // Either new file or the file has changed
   if (cf && (strlen(cf) > 0)) {
      // The file must exist and be readable
      if (gSystem->GetPathInfo(cf, st)) {
         Error("ReadGroupConfig", "could not stat %s", cf);
         return kFALSE;
      }
      if (gSystem->AccessPathName(cf, kReadPermission)) {
         Error("ReadGroupConfig", "cannot read %s", cf);
         return kFALSE;
      }
      // Ok
      fGroupConfigFile = cf;
      fMTimeGroupConfig = st.fMtime;
   }

   if (gDebug > 0)
      Info("ReadGroupConfig","reading group config from %s", cf);

   // Open the config file
   ifstream in;
   in.open(cf);
   if (!in.is_open()) {
      Error("ReadGroupConfig", "could not open config file %s", cf);
      return kFALSE;
   }

   // Container for the global common user
   TString tmpCommonUser;

   // Go through
   TString line;
   while (in.good()) {
      // Read new line
      line.ReadLine(in);
      // Parse it
      Ssiz_t from = 0;
      TString key;
      if (!line.Tokenize(key, from, " ")) // No token
         continue;
      // Parsing depends on the key
      if (key == "property") {
         // Read group
         TString grp;
         if (!line.Tokenize(grp, from, " ")) {// No token
            if (gDebug > 0)
               Info("ReadGroupConfig","incomplete line: '%s'", line.Data());
            continue;
         }
         // Read type of property
         TString type;
         if (!line.Tokenize(type, from, " ")) // No token
            continue;
         if (type == "diskquota") {
            // Read diskquota
            TString sdq;
            if (!line.Tokenize(sdq, from, " ")) // No token
               continue;
            // Enforce GigaBytes as default
            if (sdq.IsDigit()) sdq += "G";
            Long64_t quota = ToBytes(sdq);
            if (quota > -1) {
               fGroupQuota.Add(new TObjString(grp),
                               new TParameter<Long64_t> ("group quota", quota));
            } else {
               Warning("ReadGroupConfig",
                       "problems parsing string: wrong or unsupported suffix? %s",
                        sdq.Data());
            }
         } else if (type == "commonuser") {
            // Read common user for this group
            TString comusr;
            if (!line.Tokenize(comusr, from, " ")) // No token
               continue;

         }

      } else if (key == "dataset") {
         // Read type
         TString type;
         if (!line.Tokenize(type, from, " ")) {// No token
            if (gDebug > 0)
               Info("ReadGroupConfig","incomplete line: '%s'", line.Data());
            continue;
         }
         if (type == "commonuser") {
            // Read global common user
            TString comusr;
            if (!line.Tokenize(comusr, from, " ")) // No token
               continue;
            fCommonUser = comusr;
         } else if (type == "commongroup") {
            // Read global common group
            TString comgrp;
            if (!line.Tokenize(comgrp, from, " ")) // No token
               continue;
            fCommonGroup = comgrp;
         } else if (type == "diskquota") {
            // Quota check switch
            TString on;
            if (!line.Tokenize(on, from, " ")) // No token
               continue;
            if (on == "on") {
               SetBit(TDataSetManager::kCheckQuota);
            } else if (on == "off") {
               ResetBit(TDataSetManager::kCheckQuota);
            }
         }

      } else if (key == "averagefilesize") {

         // Read average size
         TString avgsize;
         if (!line.Tokenize(avgsize, from, " ")) {// No token
            if (gDebug > 0)
               Info("ReadGroupConfig","incomplete line: '%s'", line.Data());
            continue;
         }
         Long64_t avgsz = ToBytes(avgsize);
         if (avgsz > -1) {
            fAvgFileSize = avgsz;
         } else {
            Warning("ReadGroupConfig",
                    "problems parsing string: wrong or unsupported suffix? %s",
                    avgsize.Data());
         }
      }
   }
   in.close();

   return kTRUE;
}

//______________________________________________________________________________
Long64_t TDataSetManager::ToBytes(const char *size)
{
   // Static utility function to gt the number of bytes from a string
   // representation in the form "<digit><sfx>" with <sfx> = {"", "k", "M", "G",
   // "T", "P"} (case insensitive).
   // Returns -1 if the format is wrong.

   Long64_t lsize = -1;

   // Check if valid
   if (!size || strlen(size) <= 0) return lsize;

   TString s(size);
   // Determine factor
   Long64_t fact = 1;
   if (!s.IsDigit()) {
      const char *unit[5] = { "k", "M", "G", "T", "P"};
      fact = 1024;
      Int_t jj = 0;
      while (jj <= 4) {
         if (s.EndsWith(unit[jj], TString::kIgnoreCase)) {
            s.Remove(s.Length()-1);
            break;
         }
         fact *= 1024;
         jj++;
      }
   }
   // Apply factor now
   if (s.IsDigit())
      lsize = s.Atoi() * fact;

   // Done
   return lsize;
}

//______________________________________________________________________________
TFileCollection *TDataSetManager::GetDataSet(const char *, const char *)
{
   // Utility function used in various methods for user dataset upload.

   AbstractMethod("GetDataSet");
   return (TFileCollection *)0;
}

//______________________________________________________________________________
Bool_t TDataSetManager::RemoveDataSet(const char *)
{
   // Removes the indicated dataset

   AbstractMethod("RemoveDataSet");
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TDataSetManager::ExistsDataSet(const char *)
{
   // Checks if the indicated dataset exits

   AbstractMethod("ExistsDataSet");
   return kFALSE;
}

//______________________________________________________________________________
TMap *TDataSetManager::GetDataSets(const char *, UInt_t)
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

   AbstractMethod("GetDataSets");

   return (TMap *)0;
}

//______________________________________________________________________________
Int_t TDataSetManager::ScanDataSet(const char *, UInt_t)
{
   // Scans the dataset indicated by <uri> and returns the number of missing files.
   // Returns -1 if any failure occurs.
   // For more details, see documentation of
   // ScanDataSet(TFileCollection *dataset, const char *option)

   AbstractMethod("ScanDataSet");

   return -1;
}

//______________________________________________________________________________
void TDataSetManager::GetQuota(const char *group, const char *user,
                                    const char *dsName, TFileCollection *dataset)
{
   //
   // Gets quota information from this dataset

   if (gDebug > 0)
      Info("GetQuota", "processing dataset %s %s %s", group, user, dsName);

   if (dataset->GetTotalSize() > 0) {
      TParameter<Long64_t> *size =
         dynamic_cast<TParameter<Long64_t>*> (fGroupUsed.GetValue(group));
      if (!size) {
         size = new TParameter<Long64_t> ("group used", 0);
         fGroupUsed.Add(new TObjString(group), size);
      }

      size->SetVal(size->GetVal() + dataset->GetTotalSize());

      TMap *userMap = dynamic_cast<TMap*> (fUserUsed.GetValue(group));
      if (!userMap) {
         userMap = new TMap;
         fUserUsed.Add(new TObjString(group), userMap);
      }

      size = dynamic_cast<TParameter<Long64_t>*> (userMap->GetValue(user));
      if (!size) {
         size = new TParameter<Long64_t> ("user used", 0);
         userMap->Add(new TObjString(user), size);
      }

      size->SetVal(size->GetVal() + dataset->GetTotalSize());
   }
}

//______________________________________________________________________________
void TDataSetManager::ShowQuota(const char *opt)
{
   // Display quota information

   UpdateUsedSpace();

   TMap *groupQuotaMap = GetGroupQuotaMap();
   TMap *userUsedMap = GetUserUsedMap();
   if (!groupQuotaMap || !userUsedMap)
      return;

   Bool_t noInfo = kTRUE;
   TIter iter(groupQuotaMap);
   TObjString *group = 0;
   while ((group = dynamic_cast<TObjString*> (iter.Next()))) {
      noInfo = kFALSE;
      Long64_t groupQuota = GetGroupQuota(group->String());
      Long64_t groupUsed = GetGroupUsed(group->String());

      Printf(" +++ Group %s uses %.1f GB out of %.1f GB", group->String().Data(),
                                        (Float_t) groupUsed / DSM_ONE_GB,
                                       (Float_t) groupQuota / DSM_ONE_GB);

      // display also user information
      if (opt && !TString(opt).Contains("U", TString::kIgnoreCase))
         continue;

      TMap *userMap = dynamic_cast<TMap*> (userUsedMap->GetValue(group->String()));
      if (!userMap)
         continue;

      TIter iter2(userMap);
      TObjString *user = 0;
      while ((user = dynamic_cast<TObjString*> (iter2.Next()))) {
         TParameter<Long64_t> *size2 =
            dynamic_cast<TParameter<Long64_t>*> (userMap->GetValue(user->String().Data()));
         if (!size2)
            continue;

         Printf(" +++  User %s uses %.1f GB", user->String().Data(),
                                  (Float_t) size2->GetVal() / DSM_ONE_GB);
      }

      Printf("------------------------------------------------------");
   }
   // Check if something has been printed
   if (noInfo) {
      Printf(" +++ Quota check enabled but no quota info available +++ ");
   }
}

//______________________________________________________________________________
void TDataSetManager::PrintUsedSpace()
{
   //
   // Prints the quota

   Info("PrintUsedSpace", "listing used space");

   TIter iter(&fUserUsed);
   TObjString *group = 0;
   while ((group = dynamic_cast<TObjString*> (iter.Next()))) {
      TMap *userMap = dynamic_cast<TMap*> (fUserUsed.GetValue(group->String()));

      TParameter<Long64_t> *size =
         dynamic_cast<TParameter<Long64_t>*> (fGroupUsed.GetValue(group->String()));

      if (userMap && size) {
         Printf("Group %s: %lld B = %.2f GB", group->String().Data(), size->GetVal(),
                                      (Float_t) size->GetVal() / DSM_ONE_GB);

         TIter iter2(userMap);
         TObjString *user = 0;
         while ((user = dynamic_cast<TObjString*> (iter2.Next()))) {
            TParameter<Long64_t> *size2 =
               dynamic_cast<TParameter<Long64_t>*> (userMap->GetValue(user->String().Data()));
            if (size2)
               Printf("  User %s: %lld B = %.2f GB", user->String().Data(), size2->GetVal(),
                                            (Float_t) size2->GetVal() / DSM_ONE_GB);
         }

         Printf("------------------------------------------------------");
      }
   }
}

//______________________________________________________________________________
void TDataSetManager::MonitorUsedSpace(TVirtualMonitoringWriter *monitoring)
{
   //
   // Log info to the monitoring server

   Info("MonitorUsedSpace", "sending used space to monitoring server");

   TIter iter(&fUserUsed);
   TObjString *group = 0;
   while ((group = dynamic_cast<TObjString*> (iter.Next()))) {
      TMap *userMap = dynamic_cast<TMap*> (fUserUsed.GetValue(group->String()));
      TParameter<Long64_t> *size =
         dynamic_cast<TParameter<Long64_t>*> (fGroupUsed.GetValue(group->String()));

      if (!userMap || !size)
         continue;

      TList *list = new TList;
      list->SetOwner();
      list->Add(new TParameter<Long64_t>("_TOTAL_", size->GetVal()));
      Long64_t groupQuota = GetGroupQuota(group->String());
      if (groupQuota != -1)
         list->Add(new TParameter<Long64_t>("_QUOTA_", groupQuota));

      TIter iter2(userMap);
      TObjString *user = 0;
      while ((user = dynamic_cast<TObjString*> (iter2.Next()))) {
         TParameter<Long64_t> *size2 =
            dynamic_cast<TParameter<Long64_t>*> (userMap->GetValue(user->String().Data()));
         if (!size2)
            continue;
         list->Add(new TParameter<Long64_t>(user->String().Data(), size2->GetVal()));
      }

      monitoring->SendParameters(list, group->String());
      delete list;
   }
}

//______________________________________________________________________________
Long64_t TDataSetManager::GetGroupUsed(const char *group)
{
   //
   // Returns the used space of that group

   if (fgCommonDataSetTag == group)
      group = fCommonGroup;

   TParameter<Long64_t> *size =
      dynamic_cast<TParameter<Long64_t>*> (fGroupUsed.GetValue(group));
   if (!size) {
      if (gDebug > 0)
         Info("GetGroupUsed", "group %s not found", group);
      return 0;
   }

   return size->GetVal();
}

//______________________________________________________________________________
Long64_t TDataSetManager::GetGroupQuota(const char *group)
{
   //
   // returns the quota a group is allowed to have

   if (fgCommonDataSetTag == group)
      group = fCommonGroup;

   TParameter<Long64_t> *value =
      dynamic_cast<TParameter<Long64_t>*> (fGroupQuota.GetValue(group));
   if (!value) {
      if (gDebug > 0)
         Info("GetGroupQuota", "group %s not found", group);
      return 0;
   }
   return value->GetVal();
}

//______________________________________________________________________________
void TDataSetManager::UpdateUsedSpace()
{
   // updates the used space maps

   AbstractMethod("UpdateUsedSpace");
}

//______________________________________________________________________________
Int_t TDataSetManager::RegisterDataSet(const char *,
                                            TFileCollection *,
                                            const char *)
{
   // Register a dataset, perfoming quota checkings, if needed.
   // Returns 0 on success, -1 on failure

   AbstractMethod("RegisterDataSet");
   return -1;
}

//______________________________________________________________________________
TString TDataSetManager::CreateUri(const char *dsGroup, const char *dsUser,
                                        const char *dsName, const char *dsObjPath)
{
   // Creates URI for the dataset manger in the form '[[/dsGroup/]dsUser/]dsName[#dsObjPath]',
   // The optional dsObjPath can be in the form [subdir/]objname]'.

   TString uri;

   if (dsGroup && strlen(dsGroup) > 0) {
      if (dsUser && strlen(dsUser) > 0) {
         uri += Form("/%s/%s/", dsGroup, dsUser);
      } else {
         uri += Form("/%s/*/", dsGroup);
      }
   } else if (dsUser && strlen(dsUser) > 0) {
      uri += Form("%s/", dsUser);
   }
   if (dsName && strlen(dsName) > 0)
      uri += dsName;
   if (dsObjPath && strlen(dsObjPath) > 0)
      uri += Form("#%s", dsObjPath);

   // Done
   return uri;
}

//______________________________________________________________________________
Bool_t TDataSetManager::ParseUri(const char *uri,
                                      TString *dsGroup, TString *dsUser,
                                      TString *dsName, TString *dsTree,
                                      Bool_t onlyCurrent, Bool_t wildcards)
{
   // Parses a (relative) URI that describes a DataSet on the cluster.
   // The input 'uri' should be in the form '[[/group/]user/]dsname[#[subdir/]objname]',
   //  where 'objname' is the name of the object (e.g. the tree name) and the 'subdir'
   // is the directory in the file wher it should be looked for.
   // After resolving against a base URI consisting of proof://masterhost/group/user/
   // - meaning masterhost, group and user of the current session -
   // the path is checked to contain exactly three elements separated by '/':
   // group/user/dsname
   // If wildcards, '*' is allowed in group and user and dsname is allowed to be empty.
   // If onlyCurrent, only group and user of current session are allowed.
   // Only non-null parameters are filled by this function.
   // Returns kTRUE in case of success.

   // Append trailing slash if missing when wildcards are enabled
   TString uristr(uri);
   if (wildcards && uristr.Length() > 0 && !uristr.EndsWith("/"))
      uristr += '/';

   // Resolve given URI agains the base
   TUri resolved = TUri::Transform(uristr, fBase);
   if (resolved.HasQuery())
      Info ("ParseUri", "URI query part <%s> ignored", resolved.GetQuery().Data());

   TString path(resolved.GetPath());
   // Must be in the form /group/user/dsname
   Int_t pc = path.CountChar('/');
   if (pc != 3) {
      if (!TestBit(TDataSetManager::kIsSandbox)) {
         Error ("ParseUri", "illegal dataset path: %s", uri);
         return kFALSE;
      } else if (pc >= 0 && pc < 3) {
         // Add missing slashes
         TString sls("/");
         if (pc == 2) {
            sls = "/";
         } else if (pc == 1) {
            sls = Form("/%s/", fGroup.Data());
         } else if (pc == 0) {
            sls = Form("/%s/%s/", fGroup.Data(), fUser.Data());
         }
         path.Insert(0, sls);
      }
   }
   if (gDebug > 1)
      Info("ParseUri", "path: '%s'", path.Data());

   // Get individual values from tokens
   Int_t from = 1;
   TString group, user, name;
   path.Tokenize(group, from, "/");
   path.Tokenize(user, from, "/");
   path.Tokenize(name, from, "/");

   // The fragment may contain the subdir and the object name in the form '[subdir/]objname'
   TString tree = resolved.GetFragment();
   if (tree.EndsWith("/"))
      tree.Remove(tree.Length()-1);

   if (gDebug > 1)
      Info("ParseUri", "group: '%s', user: '%s', dsname:'%s', seg: '%s'",
                              group.Data(), user.Data(), name.Data(), tree.Data());

   // Check for unwanted use of wildcards
   if ((user == "*" || group == "*") && !wildcards) {
      Error ("ParseUri", "no wildcards allowed for user/group in this context");
      return kFALSE;
   }

   // dsname may only be empty if wildcards expected
   if (name.IsNull() && !wildcards) {
      Error ("ParseUri", "DataSet name is empty");
      return kFALSE;
   }

   // Construct regexp whitelist for checking illegal characters in user/group
   TPRegexp wcExp (wildcards ? "^(?:[A-Za-z0-9-]*|[*])$" : "^[A-Za-z0-9-]*$");

   // Check for illegal characters in all components
   if (!wcExp.Match(group)) {
      Error("ParseUri", "illegal characters in group");
      return kFALSE;
   }

   if (!wcExp.Match(user)) {
      Error("ParseUri", "illegal characters in user");
      return kFALSE;
   }

   if (name.Contains(TRegexp("[^A-Za-z0-9-._]"))) {
      Error("ParseUri", "illegal characters in dataset name");
      return kFALSE;
   }

   if (tree.Contains(TRegexp("[^A-Za-z0-9-/_]"))) {
      Error("ParseUri", "Illegal characters in subdir/object name");
      return kFALSE;
   }

   // Check user & group
   if (onlyCurrent && (group.CompareTo(fGroup) || user.CompareTo(fUser))) {
      Error("ParseUri", "only datasets from your group/user allowed");
      return kFALSE;
   }

   // fill parameters passed by reference, if defined
   if (dsGroup)
      *dsGroup = group;
   if (dsUser)
      *dsUser = user;
   if (dsName)
      *dsName = name;
   if (dsTree)
      *dsTree = tree;

   return kTRUE;
}

//______________________________________________________________________________
TMap *TDataSetManager::GetSubDataSets(const char *ds, const char *exclude)
{
   // Partition dataset 'ds' accordingly to the servers.
   // The returned TMap contains:
   //                <server> --> <subdataset> (TFileCollection)
   // where <subdataset> is the subset of 'ds' on <server>
   // The partitioning is done using all the URLs in the TFileInfo's, so the
   // resulting datasets are not mutually exclusive.
   // The string 'exclude' contains a comma-separated list of servers to exclude
   // from the map.

   TMap *map = (TMap *)0;

   if (!ds || strlen(ds) <= 0) {
      Info("GetDataSets", "dataset name undefined!");
      return map;
   }

   // Get the dataset
   TFileCollection *fc = GetDataSet(ds);
   if (!fc) {
      Info("GetDataSets", "could not retrieve the dataset '%s'", ds);
      return map;
   }

   // Get the subset
   if (!(map = fc->GetFilesPerServer(exclude))) {
      if (gDebug > 0)
         Info("GetDataSets", "could not get map for '%s'", ds);
   }

   // Cleanup
   delete fc;

   // Done
   return map;
}

//______________________________________________________________________________
void TDataSetManager::PrintDataSet(TFileCollection *fc, Int_t popt)
{
   // Formatted printout of the content of TFileCollection 'fc'.
   // Options in the form
   //           popt = u * 10 + f
   //     f    0 => header only, 1 => header + files
   //   when printing files
   //     u    0 => print file name only, 1 => print full URL

   if (!fc) return;

   Int_t f = popt%10;
   Int_t u = popt - 10 * f;

   Printf("+++");
   if (fc->GetTitle() && (strlen(fc->GetTitle()) > 0)) {
      Printf("+++ Dumping: %s: ", fc->GetTitle());
   } else {
      Printf("+++ Dumping: %s: ", fc->GetName());
   }
   Printf("%s", fc->ExportInfo("+++ Summary:", 1)->GetName());
   if (f == 1) {
      Printf("+++ Files:");
      Int_t nf = 0;
      TIter nxfi(fc->GetList());
      TFileInfo *fi = 0;
      while ((fi = (TFileInfo *)nxfi())) {
         if (u == 1)
            Printf("+++ %5d. %s", ++nf, fi->GetCurrentUrl()->GetUrl());
         else
            Printf("+++ %5d. %s", ++nf, fi->GetCurrentUrl()->GetFile());
      }
   }
   Printf("+++");
}

//______________________________________________________________________________
void TDataSetManager::ShowDataSets(const char *uri, const char *opt)
{
   // Prints formatted information about the dataset 'uri'.
   // The type and format of output is driven by 'opt':
   //
   //   1. opt = "server:srv1[,srv2[,srv3[,...]]]"
   //            Print info about the subsets of 'uri' on servers srv1, srv2, ...
   //   2. opt = "servers[:exclude:srv1[,srv2[,srv3[,...]]]]"
   //            Print info about the subsets of 'uri' on all servers, except
   //            the ones in the exclude list srv1, srv2, ...
   //   3. opt = <any>
   //            Print info about all datasets matching 'uri'
   //
   //   If 'opt' contains 'full:' the list of files in the datasets are also printed.
   //   In case 3. this is enabled only if 'uri' matches a single dataset.


   TFileCollection *fc = 0;
   TString o(opt);
   Int_t popt = 0;
   if (o.Contains("full:")) {
      o.ReplaceAll("full:","");
      popt = 1;
   }
   if (o.BeginsWith("server:")) {
      o.ReplaceAll("server:", "");
      TString srv;
      Int_t from = 0;
      while ((o.Tokenize(srv, from, ","))) {
         fc = GetDataSet(uri, srv.Data());
         PrintDataSet(fc, popt);
         delete fc;
      }
   } else if (o.BeginsWith("servers")) {
      o.ReplaceAll("servers", "");
      if (o.BeginsWith(":exclude:"))
         o.ReplaceAll(":exclude:", "");
      else
         o = "";
      TMap *dsmap = GetSubDataSets(uri, o.Data());
      if (dsmap) {
         TIter nxk(dsmap);
         TObject *k = 0;
         while ((k = nxk()) && (fc = (TFileCollection *) dsmap->GetValue(k))) {
            PrintDataSet(fc, popt);
         }
         delete dsmap;
      }
   } else {
      TString u(uri);
      if (!u.IsNull() && !u.Contains("*") && (fc = GetDataSet(uri))) {
         // Single dataset
         PrintDataSet(fc, 10 + popt);
         delete fc;
      } else {
         // Support for "*" or "/*"
         if (u == "*" || u == "/*" || u == "/*/" || u == "/*/*") u = "/*/*/";
         // Scan the existing datasets and print the content
         GetDataSets(u.Data(), (UInt_t)TDataSetManager::kPrint);
      }
   }

   return;
}
