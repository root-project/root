// @(#)root/base:$Id$
// Author: Dario Berzano, 26.11.12

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSetManagerAliEn                                                 //
//                                                                      //
// Implementation of TDataSetManager dynamically creating datasets      //
// by querying the AliEn file catalog. Retrieved information is cached. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDataSetManagerAliEn.h"
#include "TError.h"

ClassImp(TAliEnFind);

//______________________________________________________________________________
TAliEnFind::TAliEnFind(const TString &basePath, const TString &fileName,
  const TString &anchor, const Bool_t archSubst, const TString &treeName,
  const TString &regexp, const TString &query) :
  fBasePath(basePath), fFileName(fileName), fTreeName(treeName),
  fRegexpRaw(regexp), fAnchor(anchor), fQuery(query), fArchSubst(archSubst),
  fRegexp(0), fSearchId(""), fGridResult(0)
{
   // Constructor

   if (fArchSubst) fAnchor = ""; // no anchor when substituting zipfile
   if (!fRegexpRaw.IsNull())
      fRegexp = new TPMERegexp(regexp);
}

//______________________________________________________________________________
TAliEnFind::TAliEnFind(const TAliEnFind &src) : TObject()
{
   // Copy constructor. Cached query result is not copied

   fBasePath = src.fBasePath;
   fFileName = src.fFileName;
   fAnchor = src.fAnchor;
   fArchSubst = src.fArchSubst;
   fQuery = src.fQuery;
   fTreeName = src.fTreeName;
   fRegexpRaw = src.fRegexpRaw;

   if (src.fRegexp)
      fRegexp = new TPMERegexp( *(src.fRegexp) );
   else
      fRegexp = NULL;

   fGridResult = NULL;
}

//______________________________________________________________________________
TAliEnFind &TAliEnFind::operator=(const TAliEnFind &rhs)
{
   // Assignment operator. Cached query result is not copied

   if (&rhs != this) {
      fBasePath = rhs.fBasePath;
      fFileName = rhs.fFileName;
      fAnchor = rhs.fAnchor;
      fArchSubst = rhs.fArchSubst;
      fQuery = rhs.fQuery;
      fTreeName = rhs.fTreeName;

      SetRegexp(rhs.fRegexpRaw);

      InvalidateSearchId();
      InvalidateGridResult();
   }

   return *this;
}

//______________________________________________________________________________
TAliEnFind::~TAliEnFind()
{
   // Destructor

   if (fRegexp) delete fRegexp;
   if (fGridResult) delete fGridResult;
}

//______________________________________________________________________________
TGridResult *TAliEnFind::GetGridResult(Bool_t forceNewQuery)
{
   // Query the AliEn file catalog

   if (fGridResult && !forceNewQuery) {
      if (gDebug >= 1)
         Info("GetGridResult", "Returning cached AliEn find results");
      return fGridResult;
   }
   else if (gDebug >= 1) {
      Info("GetGridResult", "Querying AliEn file catalog");
   }

   InvalidateGridResult();

   if (!gGrid) {
      TGrid::Connect("alien:");
      if (!gGrid) return NULL;
   }

   if (gDebug >= 1) {
      Info("GetGridResult", "AliEn find %s %s [regexp=%s] [archsubst=%d]",
         fBasePath.Data(), fFileName.Data(), fRegexpRaw.Data(), fArchSubst);
   }

   fGridResult = gGrid->Query(fBasePath.Data(), fFileName.Data());
   if (!fGridResult) return NULL;

   if (fRegexp || fArchSubst || !fAnchor.IsNull() || !fQuery.IsNull()) {

      TPMERegexp *reArchSubst = NULL;
      TString substWith;
      if (fArchSubst) {
         TString temp;
         temp.Form("/%s$", fFileName.Data());
         reArchSubst = new TPMERegexp(temp.Data());
         if (fQuery) {
            substWith.Form("/root_archive.zip?%s#%s", fQuery.Data(),
               fFileName.Data());
         }
         else {
            substWith.Form("/root_archive.zip#%s", fFileName.Data());
         }
      }

      TIter it(fGridResult);
      TMap *map;
      TObjString *os;
      TString tUrl;

      while (( map = dynamic_cast<TMap *>(it.Next()) ) != NULL) {

         os = dynamic_cast<TObjString *>( map->GetValue("turl") );
         if (!os) continue;
         tUrl = os->String();

         if (fRegexp && (fRegexp->Match(tUrl) == 0)) {
            // Remove object if it does not match expression
            TObject *exmap = fGridResult->Remove(map);
            if (exmap) delete exmap;  // Remove() does not delete
         }

         if (reArchSubst) {
            // Substitute file_name with containing_archive.zip#file_name
            reArchSubst->Substitute(tUrl, substWith, kFALSE);
            os->SetString(tUrl.Data());
         }
         else if (!fAnchor.IsNull()) {
            // Append anchor (and, possibly, query first)
            if (!fQuery.IsNull()) {
               tUrl.Append("?");
               tUrl.Append(fQuery);
            }
            tUrl.Append("#");
            tUrl.Append(fAnchor);
            os->SetString(tUrl.Data());
         }
         else if (!fQuery.IsNull()) {
            // Append query only
            tUrl.Append("?");
            tUrl.Append(fQuery);
            os->SetString(tUrl.Data());
         }
      }

      if (reArchSubst) delete reArchSubst;
   }

   return fGridResult;
}

//______________________________________________________________________________
const char *TAliEnFind::GetSearchId()
{
  if (fSearchId.IsNull()) {
    TString searchIdStr;
    searchIdStr.Form("BasePath=%s FileName=%s Anchor=%s ArchSubst=%d "
      "TreeName=%s Regexp=%s",
      fBasePath.Data(), fFileName.Data(), fAnchor.Data(), fArchSubst,
      fTreeName.Data(), fRegexpRaw.Data());
    TMD5 *md5 = new TMD5();
    md5->Update( (const UChar_t *)searchIdStr.Data(),
      (UInt_t)searchIdStr.Length() );
    md5->Final();
    fSearchId = md5->AsString();
    delete md5;
  }
  if (gDebug >= 2)
    Info("GetSearchId", "Returning search ID %s", fSearchId.Data());
  return fSearchId.Data();
}

//______________________________________________________________________________
TFileCollection *TAliEnFind::GetCollection(Bool_t forceNewQuery)
{
  GetGridResult(forceNewQuery);
  if (!fGridResult) return NULL;

  Int_t nEntries = fGridResult->GetEntries();
  TFileCollection *fc = new TFileCollection();

  for (Int_t i=0; i<nEntries; i++) {

    Long64_t size = TString(fGridResult->GetKey(i, "size")).Atoll();

    TString tUrl = fGridResult->GetKey(i, "turl");

    if (gDebug >= 2)
      Info("GetCollection", ">> %s", tUrl.Data());

    // Append to TFileCollection: url, size, guid, md5
    TFileInfo *fi = new TFileInfo( tUrl, size, fGridResult->GetKey(i, "guid"),
      fGridResult->GetKey(i, "md5") );

    fc->Add(fi);

  }

  if (fTreeName != "")
    fc->SetDefaultTreeName(fTreeName.Data());

  fc->Update();  // needed for summary info

  return fc;
}

//______________________________________________________________________________
void TAliEnFind::Print(Option_t* opt) const
{
   if (opt) {}  // silence warning
   Printf("BasePath=%s FileName=%s Anchor=%s ArchSubst=%d "
      "TreeName=%s Regexp=%s (query %s a result)",
      fBasePath.Data(), fFileName.Data(), fAnchor.Data(), fArchSubst,
      fTreeName.Data(), fRegexpRaw.Data(), (fGridResult ? "has" : "has not"));
}

//______________________________________________________________________________
void TAliEnFind::SetBasePath(const char *basePath)
{
   if (fBasePath.EqualTo(basePath)) return;
   fBasePath = basePath;
   InvalidateGridResult();
   InvalidateSearchId();
}

//______________________________________________________________________________
void TAliEnFind::SetFileName(const char *fileName)
{
   if (fFileName.EqualTo(fileName)) return;
   fFileName = fileName;
   InvalidateGridResult();
   InvalidateSearchId();
}

//______________________________________________________________________________
void TAliEnFind::SetAnchor(const char *anchor)
{
   if (fAnchor.EqualTo(anchor)) return;
   fAnchor = anchor;
   InvalidateGridResult();
   InvalidateSearchId();
}

//______________________________________________________________________________
void TAliEnFind::SetTreeName(const char *treeName)
{
   if (fTreeName.EqualTo(treeName)) return;
   fTreeName = treeName;
   InvalidateSearchId();
}

//______________________________________________________________________________
void TAliEnFind::SetArchSubst(Bool_t archSubst)
{
   if (fArchSubst == archSubst) return;
   fArchSubst = archSubst;
   InvalidateGridResult();
   InvalidateSearchId();
}

//______________________________________________________________________________
void TAliEnFind::SetRegexp(const char *regexp)
{
   if (fRegexpRaw.EqualTo(regexp)) return;

   fRegexpRaw = regexp;
   if (fRegexp) delete fRegexp;
   if (!fRegexpRaw.IsNull())
      fRegexp = new TPMERegexp(regexp);
   else
      fRegexp = NULL;

   InvalidateGridResult();
   InvalidateSearchId();
}

//______________________________________________________________________________
void TAliEnFind::InvalidateSearchId()
{
   if (!fSearchId.IsNull())
      fSearchId = "";
}

//______________________________________________________________________________
void TAliEnFind::InvalidateGridResult()
{
   if (fGridResult) {
      delete fGridResult;
      fGridResult = NULL;
   }
}

ClassImp(TDataSetManagerAliEn);

//______________________________________________________________________________
void TDataSetManagerAliEn::Init(TString cacheDir, TString urlTpl,
  ULong_t cacheExpire_s)
{
  kfNoopRedirUrl = new TUrl("noop://redir");
  kfNoopUnknownUrl = new TUrl("noop://unknown");
  kfNoopNoneUrl = new TUrl("noop://none");

  fCacheExpire_s = cacheExpire_s;
  fUrlRe = new TPMERegexp("^alien://(.*)$");
  fUrlTpl = urlTpl;

  if (fUrlTpl.Contains("<path>")) {
    // Ordinary pattern, something like root://host/prefix/<path>
    fUrlTpl.ReplaceAll("<path>", "$1");
  }
  else {
    // No <path> to substitute: assume it is a SE (storage element) name
    fReadFromSE = kTRUE;
  }

  TString dsDirFmt;
  dsDirFmt.Form("dir:%s perms:open", cacheDir.Data());
  fCache = new TDataSetManagerFile("_cache_", "_cache_", dsDirFmt);

  if (fCache->TestBit(TObject::kInvalidObject)) {
    Error("Init", "Cannot initialize cache on directory %s", cacheDir.Data());
    SetBit(TObject::kInvalidObject);
    return;
  }

  // Provided for compatibility
  ResetBit(TDataSetManager::kAllowRegister);  // impossible to register
  ResetBit(TDataSetManager::kCheckQuota);  // quota control off

  if (gDebug >= 1) {
    Info("TDataSetManagerAliEn", "Caching on %s", cacheDir.Data());
    Info("TDataSetManagerAliEn", "URL schema: %s", urlTpl.Data());
    Info("TDataSetManagerAliEn", "Cache expires after: %lus", cacheExpire_s);
  }
}

//______________________________________________________________________________
TDataSetManagerAliEn::TDataSetManagerAliEn(const char *cacheDir,
  const char *urlTpl, ULong_t cacheExpire_s)
  : TDataSetManager("", "", ""), fUrlRe(0), fCache(0), fReadFromSE(kFALSE),
    kfNoopRedirUrl(0), kfNoopUnknownUrl(0), kfNoopNoneUrl(0)
{
  Init(cacheDir, urlTpl, cacheExpire_s);
}

//______________________________________________________________________________
TDataSetManagerAliEn::TDataSetManagerAliEn(const char *, const char *,
  const char *cfgStr) : TDataSetManager("", "", ""), fUrlRe(0), fCache(0),
  fReadFromSE(kFALSE), kfNoopRedirUrl(0), kfNoopUnknownUrl(0), kfNoopNoneUrl(0)
{
  // Compatibility with the plugin manager

  TPMERegexp reCache("(^| )cache:([^ ]+)( |$)");
  if (reCache.Match(cfgStr) != 4) {
    Error("TDataSetManagerAliEn", "No cache directory specified");
    SetBit(TObject::kInvalidObject);
    return;
  }

  TPMERegexp reUrlTpl("(^| )urltemplate:([^ ]+)( |$)");
  if (reUrlTpl.Match(cfgStr) != 4) {
    Error("TDataSetManagerAliEn", "No local URL template specified");
    SetBit(TObject::kInvalidObject);
    return;
  }

  TPMERegexp reCacheExpire("(^| )cacheexpiresecs:([0-9]+)( |$)");
  if (reCacheExpire.Match(cfgStr) != 4) {
    Error("TDataSetManagerAliEn", "No cache expiration set");
    SetBit(TObject::kInvalidObject);
    return;
  }

  Init(reCache[2], reUrlTpl[2], (ULong_t)reCacheExpire[2].Atoll());
}

//______________________________________________________________________________
TDataSetManagerAliEn::~TDataSetManagerAliEn()
{
  if (fCache) delete fCache;
  if (fUrlRe) delete fUrlRe;
  if (kfNoopRedirUrl) delete kfNoopRedirUrl;
  if (kfNoopUnknownUrl) delete kfNoopUnknownUrl;
  if (kfNoopNoneUrl) delete kfNoopNoneUrl;
}

//______________________________________________________________________________
TList *TDataSetManagerAliEn::GetFindCommandsFromUri(TString &uri,
  EDataMode &dataMode, Bool_t &forceUpdate)
{
  // Parse kind
  TPMERegexp reKind("^(Data;|Sim;|Find;)");
  if (reKind.Match(uri) != 2) {
    Error("GetFindCommandsFromUri", "Data, Sim or Find not specified");
    return NULL;
  }

  // Parse data mode (remote, local, cache -- optional)
  TPMERegexp reMode("(^|;)Mode=([A-Za-z]+)(;|$)");
  if (reMode.Match(uri) != 4) {
    dataMode = kDataLocal;  // default
  }
  else {
    if (reMode[2].EqualTo("remote", TString::kIgnoreCase))
      dataMode = kDataRemote;
    else if (reMode[2].EqualTo("local", TString::kIgnoreCase))
      dataMode = kDataLocal;
    else if (reMode[2].EqualTo("cache", TString::kIgnoreCase))
      dataMode = kDataCache;
    else {
      Error("GetFindCommandsFromUri",
        "Wrong analysis mode specified: use one of: Mode=remote, local, cache");
      return NULL;  // no mode is ok, but wrong mode is not
    }
  }

  TList *findCommands = NULL;

  if (reKind[1].BeginsWith("Find")) {

    TString basePath;
    TString fileName;
    TString anchor;
    TString query;
    TString treeName;
    TString regexp;

    // Custom search URI
    if (!ParseCustomFindUri(uri, basePath, fileName, anchor, query, treeName,
      regexp)) {
      Error("GetFindCommandsFromUri", "Malformed AliEn find command");
      return NULL;
    }

    findCommands = new TList();
    findCommands->SetOwner();
    findCommands->Add( new TAliEnFind(basePath, fileName, anchor, kFALSE,
      treeName, regexp, query) );

  }
  else {  // Data or Sim
    Bool_t sim = (reKind[1][0] == 'S');
    TString lhcPeriod;
    Int_t year;
    std::vector<Int_t> *runList;
    Bool_t esd;
    Int_t aodNum;
    TString pass;

    if (!ParseOfficialDataUri(uri, sim, lhcPeriod, year, runList, esd,
      aodNum, pass)) {
      Error("GetFindCommandsFromUri", "Invalid parameters");
      return NULL;
    }

    findCommands = new TList();
    findCommands->SetOwner(kTRUE);

    TString basePathRun;

    if (!gGrid) {
      TGrid::Connect("alien:");
      if (!gGrid) {
        delete findCommands;
        delete runList;
        return NULL;
      }
    }

    if (sim) {
      // Montecarlo init.
      // Check whether this period is in /alice/sim/<period> or in
      // /alice/sim/<year>/<period> and act properly, since naming convention
      // is unclear!

      // Check once for all
      basePathRun.Form("/alice/sim/%s", lhcPeriod.Data());  // no year
      if (!gGrid->Cd(basePathRun.Data())) {
        basePathRun.Form("/alice/sim/%d/%s", year, lhcPeriod.Data());
      }
    }
    else {
      // Real data init.
      // Parse the pass string: if it starts with a number, prepend "pass"
      if ((pass[0] >= '0') && (pass[0] <= '9')) pass.Prepend("pass");
      basePathRun.Form("/alice/data/%d/%s", year, lhcPeriod.Data());
    }

    // Form a list of valid runs (to avoid unnecessary queries when run ranges
    // are specified)
    std::vector<Int_t> validRuns;
    {
      TGridResult *validRunDirs = gGrid->Ls( basePathRun.Data() );
      if (!validRunDirs) return NULL;

      TIter nrd(validRunDirs);
      TMap *dir;
      TObjString *os;
      validRuns.resize( (size_t)(validRunDirs->GetEntries()) );

      while (( dir = dynamic_cast<TMap *>(nrd()) ) != NULL) {
        os = dynamic_cast<TObjString *>( dir->GetValue("name") );
        if (!os) continue;
        Int_t run = (os->String()).Atoi();
        if (run > 0) validRuns.push_back(run);
      }
    }

    for (UInt_t i=0; i<runList->size(); i++) {

      // Check if current run is valid
      Bool_t valid = kFALSE;
      for (UInt_t j=0; j<validRuns.size(); j++) {
        if (validRuns[j] == (*runList)[i]) {
          valid = kTRUE;
          break;
        }
      }
      if (!valid) {
        //if (gDebug >=1) {
          Warning("TDataSetManagerAliEn::GetFindCommandsFromUri",
            "Avoiding unnecessary find on run %d: not found", (*runList)[i]);
        //}
        continue;
      }
      else if (gDebug >= 1) {
        Info("TDataSetManagerAliEn::GetFindCommandsFromUri", "Run found: %d", (*runList)[i]);
      }

      // Here we need to assemble the find string
      TString basePath, fileName, temp;

      if (sim) {
        // Montecarlo
        temp.Form("/%06d", runList->at(i));
        basePath = basePathRun + temp;

        if (!esd) {
          temp.Form("/AOD%03d", aodNum);
          basePath.Append(temp);
        }
      }
      else {
        // Real data
        temp.Form("/%09d/ESDs/%s", runList->at(i), pass.Data());
        basePath = basePathRun + temp;
        if (esd) {
          basePath.Append("/*.*");
        }
        else {
          temp.Form("/AOD%03d", aodNum);
          basePath.Append(temp);
        }
      }

      TString treeName;

      // File name and tree name
      if (esd) {
        fileName = "AliESDs.root";
        treeName = "/esdTree";
      }
      else {
        fileName = "AliAOD.root";
        treeName = "/aodTree";
      }

      findCommands->Add( new TAliEnFind(basePath, fileName, "", kTRUE,
        treeName) );

    }

    delete runList;

  }

  // Force update or use cache (when possible)
  TPMERegexp reForceUpdate("(^|;)ForceUpdate(;|$)");
  forceUpdate = (reForceUpdate.Match(uri) == 3);

  // If no valid data was found, then findCommands is NULL
  return findCommands;
}

//______________________________________________________________________________
Bool_t TDataSetManagerAliEn::ParseCustomFindUri(TString &uri,
   TString &basePath, TString &fileName, TString &anchor, TString &query,
   TString &treeName, TString &regexp)
{

  // Copy URI to a dummy URI parsed to look for unrecognized stuff; initial
  // part is known ("Find;") and stripped
  TString checkUri = uri(5, uri.Length());

  // Mode and ForceUpdate (strip them from the check string)
  TPMERegexp reMode("(^|;)(Mode=[A-Za-z]+)(;|$)");
  if (reMode.Match(uri) == 4)
    checkUri.ReplaceAll(reMode[2], "");
  TPMERegexp reForceUpdate("(^|;)(ForceUpdate)(;|$)");
  if (reForceUpdate.Match(uri) == 4)
    checkUri.ReplaceAll(reForceUpdate[2], "");

  // Base path
  TPMERegexp reBasePath("(^|;)(BasePath=([^; ]+))(;|$)");
  if (reBasePath.Match(uri) != 5) {
    ::Error("TDataSetManagerAliEn::ParseCustomFindUri",
      "Base path not specified");
    return kFALSE;
  }
  checkUri.ReplaceAll(reBasePath[2], "");
  basePath = reBasePath[3];

  // File name
  TPMERegexp reFileName("(^|;)(FileName=([^; ]+))(;|$)");
  if (reFileName.Match(uri) != 5) {
    ::Error("TDataSetManagerAliEn::ParseCustomFindUri",
      "File name not specified");
    return kFALSE;
  }
  checkUri.ReplaceAll(reFileName[2], "");
  fileName = reFileName[3];

  // Anchor (optional)
  TPMERegexp reAnchor("(^|;)(Anchor=([^; ]+))(;|$)");
  if (reAnchor.Match(uri) != 5)
    anchor = "";
  else {
    checkUri.ReplaceAll(reAnchor[2], "");
    anchor = reAnchor[3];
  }

  // Query string (optional)
  TPMERegexp reQuery("(^|;)(Query=([^; ]+))(;|$)");
  if (reQuery.Match(uri) != 5)
    query = "";
  else {
    checkUri.ReplaceAll(reQuery[2], "");
    query = reQuery[3];
  }

  // Tree name (optional)
  TPMERegexp reTreeName("(^|;)(Tree=(/[^; ]+))(;|$)");
  if (reTreeName.Match(uri) != 5)
    treeName = "";
  else {
    checkUri.ReplaceAll(reTreeName[2], "");
    treeName = reTreeName[3];
  }

  // Regexp (optional)
  TPMERegexp reRegexp("(^|;)(Regexp=([^; ]+))(;|$)");
  if (reRegexp.Match(uri) != 5)
    regexp = "";
  else {
    checkUri.ReplaceAll(reRegexp[2], "");
    regexp = reRegexp[3];
  }

  // Check for unparsed stuff; parsed stuff has been stripped from checkUri
  checkUri.ReplaceAll(";", "");
  checkUri.ReplaceAll(" ", "");
  if (!checkUri.IsNull()) {
    ::Error("TDataSetManagerAliEn::ParseCustomFindUri",
      "There are unrecognized parameters in the dataset find string");
    return kFALSE;
  }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDataSetManagerAliEn::ParseOfficialDataUri(TString &uri, Bool_t sim,
  TString &period, Int_t &year, std::vector<Int_t> *&runList, Bool_t &esd,
  Int_t &aodNum, TString &pass)
{

  // Copy URI to a dummy URI parsed to look for unrecognized stuff
  TString checkUri;

  // Strip the initial part (either "Data;" or "Sim;")
  {
    Ssiz_t idx = uri.Index(";");
    checkUri = uri(idx, uri.Length());
  }

  // Mode and ForceUpdate (strip them from the check string)
  TPMERegexp reMode("(^|;)(Mode=[A-Za-z]+)(;|$)");
  if (reMode.Match(uri) == 4)
    checkUri.ReplaceAll(reMode[2], "");
  TPMERegexp reForceUpdate("(^|;)(ForceUpdate)(;|$)");
  if (reForceUpdate.Match(uri) == 4)
    checkUri.ReplaceAll(reForceUpdate[2], "");

  //
  // Parse LHC period
  //

  TPMERegexp rePeriod("(^|;)(Period=(LHC([0-9]{2})[^;]*))(;|$)");
  if (rePeriod.Match(uri) != 6) {
    ::Error("TDataSetManagerAliEn::ParseOfficialDataUri",
      "LHC period not specified (e.g. Period=LHC10h)");
    return kFALSE;
  }

  checkUri.ReplaceAll(rePeriod[2], "");
  period = rePeriod[3];
  year = rePeriod[4].Atoi() + 2000;

  //
  // Parse data format (ESDs or AODXXX)
  //

  TPMERegexp reFormat("(^|;)(Variant=(ESDs?|AOD([0-9]{3})))(;|$)");
  if (reFormat.Match(uri) != 6) {
    ::Error("TDataSetManagerAliEn::ParseOfficialDataUri",
      "Data variant (e.g., Variant=ESD or AOD079) not specified");
    return kFALSE;
  }

  checkUri.ReplaceAll(reFormat[2], "");
  if (reFormat[3].BeginsWith("ESD")) esd = kTRUE;
  else {
    esd = kFALSE;
    aodNum = reFormat[4].Atoi();
  }

  //
  // Parse pass: mandatory on Data, useless on Sim
  //

  TPMERegexp rePass("(^|;)(Pass=([a-zA-Z_0-9-]+))(;|$)");
  if ((!sim) && (rePass.Match(uri) != 5)) {
    ::Error("TDataSetManagerAliEn::ParseOfficialDataUri",
      "Pass (e.g., Pass=cpass1_muon) is mandatory on real data");
    return kFALSE;
  }
  checkUri.ReplaceAll(rePass[2], "");
  pass = rePass[3];

  //
  // Parse run list
  //

  TPMERegexp reRun("(^|;)(Run=([0-9,-]+))(;|$)");
  if (reRun.Match(uri) != 5) {
    ::Error("TDataSetManagerAliEn::ParseOfficialDataUri",
      "Run or run range not specified (e.g., Run=139104-139107,139306)");
    return kFALSE;
  }
  checkUri.ReplaceAll(reRun[2], "");
  TString runListStr = reRun[3];
  runList = ExpandRunSpec(runListStr);  // must be freed by caller

  // Check for unparsed stuff; parsed stuff has been stripped from checkUri
  checkUri.ReplaceAll(";", "");
  checkUri.ReplaceAll(" ", "");
  if (!checkUri.IsNull()) {
    ::Error("TDataSetManagerAliEn::ParseOfficialDataUri",
      "There are unrecognized parameters in dataset string");
    return kFALSE;
  }

  return kTRUE;
}

//______________________________________________________________________________
TUrl *TDataSetManagerAliEn::AliEnWhereIs(TUrl *alienUrl, TString &closeSE,
  Bool_t onlyFromCloseSE) {

  // Performs an AliEn "whereis -r" on the given input AliEn URL. The input URL
  // is assumed to be an AliEn one, with alien:// as protocol. The "whereis"
  // command returns the full list of XRootD URLs actually pointing to the files
  // (the PFNs). The "-r" switch resolves pointers to files in archives to their
  // PFNs.
  // With closeSE a "close storage element" can be specified (like
  // "ALICE::Torino::SE"): if onlyFromCloseSE is kTRUE, the endpoint URL will be
  // returned only if there is one endpoint matching that SE (NULL is returned
  // otherwise). Elsewhere, the first URL found is returned.
  // This function might return NULL if it does not find a suitable endpoint URL
  // for the given file.

  if (!alienUrl) {
    ::Error("TDataSetManagerAliEn::AliEnWhereIs", "input AliEn URL not given!");
    return NULL;
  }

  if (!gGrid || (strcmp(gGrid->GetGrid(), "alien") != 0)) {
    ::Error("TDataSetManagerAliEn::AliEnWhereIs", "no AliEn grid connection available!");
    return NULL;
  }

  TString cmd = "whereis -r ";
  cmd.Append(alienUrl->GetFile());
  TList *resp;

  resp = dynamic_cast<TList *>( gGrid->Command(cmd.Data()) );
  if (!resp) {
    ::Error("TDataSetManagerAliEn::AliEnWhereIs", "cannot get response from AliEn");
    return NULL;
  }

  TIter nextPfn(resp);
  TMap *pfn;
  TString se, pfnUrl, validPfnUrl;
  while ( (pfn = dynamic_cast<TMap *>( nextPfn() )) != NULL ) {

    if ((pfn->GetValue("se") == NULL) || (pfn->GetValue("pfn") == NULL)) {
      continue;  // skip invalid result
    }

    pfnUrl = pfn->GetValue("pfn")->GetName();
    se = pfn->GetValue("se")->GetName();

    if (se.EqualTo(closeSE, TString::kIgnoreCase)) {
      // Found matching URL from the preferred SE
      validPfnUrl = pfnUrl;
      break;
    }
    else if (!onlyFromCloseSE && validPfnUrl.IsNull()) {
      validPfnUrl = pfnUrl;
    }

    // TIter nextPair(pfn);
    // TObjString *keyos;
    // while ( (keyos = dynamic_cast<TObjString *>( nextPair() )) != NULL ) {
    //   const char *key = keyos->GetName();
    //   const char *val = pfn->GetValue(key)->GetName();
    //   ::Info("TDataSetManagerAliEn::AliEnWhereIs", "%s-->%s", key, val);
    //   // se, pfn, guid
    // }

  }

  delete resp;

  if (validPfnUrl.IsNull()) {
    if (gDebug >= 1) {
      ::Error("TDataSetManagerAliEn::AliEnWhereIs", "cannot find endpoint URL for %s", alienUrl->GetUrl());
    }
    return NULL;
  }

  TUrl *pfnTUrl = new TUrl( validPfnUrl.Data() );

  // Append original options and the zip=<anchor> if applicable (needed!)
  TString options = alienUrl->GetOptions();
  TString anchor = alienUrl->GetAnchor();
  if (!anchor.IsNull()) {
    options.Append("&zip=");
    options.Append(anchor);
    pfnTUrl->SetAnchor(anchor.Data());
    pfnTUrl->SetOptions(options.Data());
  }

  return pfnTUrl;
}

//______________________________________________________________________________
std::vector<Int_t> *TDataSetManagerAliEn::ExpandRunSpec(TString &runSpec) {

  std::vector<Int_t> *runNumsPtr = new std::vector<Int_t>();
  std::vector<Int_t> &runNums = *runNumsPtr;

  TObjArray *runs = runSpec.Tokenize(":,");
  runs->SetOwner();
  TIter run(runs);
  TObjString *runOs;

  while ( (runOs = dynamic_cast<TObjString *>(run.Next())) ) {

    TString runStr = runOs->String();

    TPMERegexp p("^([0-9]+)-([0-9]+)$");
    if (p.Match(runStr) == 3) {
      Int_t r1 = p[1].Atoi();
      Int_t r2 = p[2].Atoi();

      if (r1 > r2) {
        // Swap
        r1 = r1 ^ r2;
        r2 = r1 ^ r2;
        r1 = r1 ^ r2;
      }

      for (Int_t r=r1; r<=r2; r++) {
        runNums.push_back(r);
      }
    }
    else {
      runNums.push_back(runStr.Atoi());
    }
  }

  delete runs;

  // Bubble sort (slow)
  for (UInt_t i=0; i<runNums.size(); i++) {
    for (UInt_t j=i+1; j<runNums.size(); j++) {
      if (runNums[j] < runNums[i]) {
        runNums[i] = runNums[i] ^ runNums[j];
        runNums[j] = runNums[i] ^ runNums[j];
        runNums[i] = runNums[i] ^ runNums[j];
      }
    }
  }

  // Remove duplicates
  {
    std::vector<Int_t>::iterator itr = runNums.begin();
    Int_t prevVal = 0;  // unneeded but silences uninitialized warning
    while (itr != runNums.end()) {
      if ((itr == runNums.begin()) || (prevVal != *itr)) {
        prevVal = *itr;
        itr++;
      }
      else {
        itr = runNums.erase(itr);
      }
    }
  }

  return runNumsPtr;

}

//______________________________________________________________________________
TFileCollection *TDataSetManagerAliEn::GetDataSet(const char *uri, const char *)
{
  TFileCollection *fc = NULL;  // global collection

  TString sUri(uri);
  EDataMode dataMode;
  Bool_t forceUpdate;
  TList *findCmds = GetFindCommandsFromUri(sUri, dataMode, forceUpdate);
  if (!findCmds) return NULL;

  fc = new TFileCollection();  // this fc will contain all data

  TFileStager *fstg = NULL; // used and reused for bulk lookup
  TFileInfo *fi;

  TIter it(findCmds);
  TAliEnFind *af;
  while ((af = dynamic_cast<TAliEnFind *>(it.Next())) != NULL) {

    TString cachedUri = af->GetSearchId();
    TFileCollection *newFc = NULL;
    Bool_t saveToCache = kFALSE;
    Bool_t fillLocality = kFALSE;

    // Check modified time
    Long_t mtime = fCache->GetModTime(cachedUri.Data());
    Long_t now = gSystem->Now();
    now = now/1000 + 788914800;  // magic is secs between Jan 1st 1970 and 1995

    if (forceUpdate) {
      if (gDebug >= 1)
        Info("GetDataSet", "Ignoring cached query result: forcing update");
    }
    else if ((mtime > 0) && (now-mtime > fCacheExpire_s)) {
      if (gDebug >= 1)
        Info("GetDataSet", "Dataset cache has expired");
    }
    else {
      if (gDebug >= 1)
        Info("GetDataSet", "Getting file collection from cache");
      newFc = fCache->GetDataSet(cachedUri.Data());
    }

    if (!newFc) {

      if (gDebug >= 1)
        Info("GetDataSet", "Getting file collection from AliEn");

      newFc = af->GetCollection();
      if (!newFc) {
        Error("GetDataSet", "Cannot get collection from AliEn");
        delete findCmds;
        delete fc;
        return NULL;
      }

      // Dataset was not cached. Just got from AliEn. Either fill with endpoint,
      // if kDataLocal, or fill with dummy data, if kDataRemote/kDataLocal.
      // Inside this scope we are processing data that will be cached, and not
      // data actually returned to the user

      // Add redirector's URL
      TIter itCache(newFc->GetList());
      TString tUrl;
      while ((fi = dynamic_cast<TFileInfo *>(itCache.Next()))) {

        if (fReadFromSE) {
          TUrl *seUrl;
          seUrl = AliEnWhereIs( fi->GetCurrentUrl(), fUrlTpl, kTRUE );

          if (seUrl) {
            // File is present (according to catalog) in the given SE
            fi->AddUrl(seUrl->GetUrl(), kTRUE);  // kTRUE == prepend URL
            fi->ResetUrl();
            delete seUrl;
          }
          else {
            // File not found in that SE
            fi->AddUrl(kfNoopNoneUrl->GetUrl(), kTRUE);
          }

        }
        else {
          tUrl = fi->GetCurrentUrl()->GetUrl();
          fUrlRe->Substitute(tUrl, fUrlTpl);
          fi->AddUrl(tUrl.Data(), kTRUE);  // kTRUE == prepend URL
          fi->ResetUrl();
        }

      }

      // Add endpoint?
      if (dataMode == kDataLocal) {
        fillLocality = kTRUE;
      }
      else {
        // Don't make the user waste time: don't cache dataset locality info at
        // this time, and signal our ignorance with a dummy URL
        if (gDebug >= 1)
          Info("GetDataSet", "Not caching data locality information now");
        itCache.Reset();
        while ((fi = dynamic_cast<TFileInfo *>(itCache.Next())))
          fi->AddUrl(kfNoopUnknownUrl->GetUrl(), kTRUE);
      }

      // Update summary information and save to cache!
      saveToCache = kTRUE;

    } // end dataset just got from AliEn
    else {

      // Reading dataset from cache. Check if it has endpoint information.
      Bool_t hasEndp = kTRUE;

      fi = dynamic_cast<TFileInfo *>(newFc->GetList()->At(0));
      if (fi) {
        if ( strcmp(fi->GetCurrentUrl()->GetUrl(), kfNoopUnknownUrl->GetUrl()) == 0 ) {
          if (gDebug >= 1)
            Info("GetDataSet", "No dataset locality information in cache");
          hasEndp = kFALSE;
        }
      }

      if ((dataMode == kDataLocal) && !hasEndp) {
        // Fill missing locality information now

        // Remove first dummy URL everywhere
        TIter itCache(newFc->GetList());
        while ((fi = dynamic_cast<TFileInfo *>(itCache.Next()))) {
          fi->RemoveUrl( kfNoopUnknownUrl->GetUrl() );
        }

        fillLocality = kTRUE;  // will locate
        saveToCache = kTRUE;  // will cache
      }

    } // end processing dataset from cache

    // Fill locality: initialize stager, locate URLs
    if (fillLocality) {

      if (fReadFromSE) {

        // If we have the redirector's URL, file is staged; elsewhere, assume
        // that it is not. This way of filling locality info does not imply
        // queries. The "dummy" URL signalling that no suitable redirector is
        // there is not removed (it will be in the final results)

        TIter nxtLoc(newFc->GetList());
        while (( fi = dynamic_cast<TFileInfo *>(nxtLoc()) )) {
          if (fi->FindByUrl( kfNoopNoneUrl->GetUrl() )) {
            fi->ResetBit(TFileInfo::kStaged);
          }
          else {
            fi->SetBit(TFileInfo::kStaged);
          }
        }

      }
      else {

        // Fill locality with a redirector

        fi = dynamic_cast<TFileInfo *>(newFc->GetList()->At(0));
        if (fi) {
          Info("GetDataSet", "Filling dataset locality information: "
            "it might take time, be patient!");

          // Lazy stager initialization
          if (!fstg)
            fstg = TFileStager::Open(fi->GetCurrentUrl()->GetUrl());

          if (!fstg) {
            Error("GetDataSet", "Can't create file stager");
            delete newFc;
            delete fc;
            delete findCmds;
            return NULL;
          }
          else {
            Int_t rv = fstg->LocateCollection(newFc, kTRUE);
            if (rv < 0) {
              Error("GetDataSet", "Endpoint lookup returned an error");
              delete fstg;
              delete newFc;
              delete fc;
              delete findCmds;
              return NULL;
            }
            else if (gDebug >= 1) {
              Info("GetDataSet", "Lookup successful for %d file(s)", rv);
            }
          }
        } // end if fi
      }

    } // end if fillLocality

    // Save (back) to cache if requested
    if (saveToCache) {
      newFc->Update();
      TString group, user, name;
      fCache->ParseUri(cachedUri, &group, &user, &name);
      if (fCache->WriteDataSet(group, user, name, newFc) == 0) {
        // Non-fatal error, but warn user
        Warning("GetDataSet", "Could not cache retrieved information");
      }
    }

    // Just print the newFc (debug)
    if (gDebug >= 2) {
      Info("GetDataSet", "Dataset information currently cached follows");
      newFc->Print("filter:SsCc");
    }

    // Now we prepare the final dataset, by appending proper information from
    // newFc to fc. Cache has been already saved (possibly with locality info)

    TIter itCache(newFc->GetList());
    Int_t nDeleteUrls;
    while ((fi = dynamic_cast<TFileInfo *>(itCache.Next()))) {

      // Keep only URLs requested by user: remove the rest. We are acting on
      // the user's copy, not on the cached copy

      fi->ResetUrl();

      if (dataMode == kDataRemote) {
        // Assume remote file is always available
        fi->SetBit(TFileInfo::kStaged);
        // Only last URL should survive
        nDeleteUrls = fi->GetNUrls() - 1;
        for (Int_t i=0; i<nDeleteUrls; i++) {
          fi->RemoveUrlAt(0);
        }
      }
      else if (dataMode == kDataCache) {
        // Access from redirector: pretend that everything is staged
        fi->SetBit(TFileInfo::kStaged);
        // Only two last URLs should survive
        nDeleteUrls = fi->GetNUrls() - 2;
        for (Int_t i=0; i<nDeleteUrls; i++) {
          fi->RemoveUrlAt(0);
        }
      }
      // else {}  // dataMode == kLocal (trust all: also the staged bit)

      // Now remove all dummy URLs
      fi->RemoveUrl( kfNoopUnknownUrl->GetUrl() );
      fi->RemoveUrl( kfNoopNoneUrl->GetUrl() );
      fi->RemoveUrl( kfNoopRedirUrl->GetUrl() );

      fi->ResetUrl();

      // Append to big file collection used for analysis
      TFileInfo *newFi = new TFileInfo(*fi);
      fc->Add(newFi);

    }

    // Set default tree
    if (!fc->GetDefaultTreeName())
      fc->SetDefaultTreeName(newFc->GetDefaultTreeName());

    delete newFc;

  } // end loop over find commands

  delete findCmds;
  if (fstg) delete fstg;

  fc->Update();
  return fc;
}

//______________________________________________________________________________
Bool_t TDataSetManagerAliEn::ExistsDataSet(const char *uri)
{
   TFileCollection *fc = GetDataSet(uri);
   Bool_t existsNonEmpty = (fc && (fc->GetNFiles() > 0));
   if (fc) delete fc;
   return existsNonEmpty;
}

//______________________________________________________________________________
Int_t TDataSetManagerAliEn::RegisterDataSet(const char *, TFileCollection *,
  const char *)
{
  MayNotUse("RegisterDataSet");
  return -1;
}

//______________________________________________________________________________
TMap *TDataSetManagerAliEn::GetDataSets(const char *, UInt_t)
{
  MayNotUse("GetDataSets");
  return NULL;
}

//______________________________________________________________________________
void TDataSetManagerAliEn::ShowDataSets(const char *, const char *)
{
  MayNotUse("ShowDataSets");
}

//______________________________________________________________________________
Bool_t TDataSetManagerAliEn::RemoveDataSet(const char *)
{
  MayNotUse("RemoveDataSet");
  return kFALSE;
}

//______________________________________________________________________________
Int_t TDataSetManagerAliEn::ScanDataSet(const char *, UInt_t)
{
  MayNotUse("ScanDataSet");
  return -1;
}

//______________________________________________________________________________
Int_t TDataSetManagerAliEn::ShowCache(const char *)
{
  MayNotUse("ShowCache");
  return -1;
}

//______________________________________________________________________________
Int_t TDataSetManagerAliEn::ClearCache(const char *)
{
  MayNotUse("ClearCache");
  return -1;
}
