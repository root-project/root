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

ClassImp(TAliEnFind);

//______________________________________________________________________________
TAliEnFind::TAliEnFind(const TString &basePath, const TString &fileName,
  const TString &anchor, const Bool_t archSubst, const TString &treeName,
  const TString &regexp) :
  fBasePath(basePath), fFileName(fileName), fTreeName(treeName),
  fRegexpRaw(regexp), fAnchor(anchor), fArchSubst(archSubst), fRegexp(0),
  fSearchId(""), fGridResult(0)
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
      Info("GetGridResult", "AliEn find %s %s [regexp=%s]",
         fBasePath.Data(), fFileName.Data(), fRegexpRaw.Data());
   }

   fGridResult = gGrid->Query(fBasePath.Data(), fFileName.Data());
   if (!fGridResult) return NULL;

   if (fRegexp || fArchSubst || (fAnchor != "")) {

      TPMERegexp *reArchSubst = NULL;
      TString substWith;
      if (fArchSubst) {
         TString temp = Form("/%s$", fFileName.Data());
         reArchSubst = new TPMERegexp(temp.Data());
         substWith = Form("/root_archive.zip#%s", fFileName.Data());
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
            reArchSubst->Substitute(tUrl, substWith, kFALSE);
            os->SetString(tUrl.Data());
         }
         else if (fAnchor) {
            tUrl.Append("#");
            tUrl.Append(fAnchor);
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
  fCacheExpire_s = cacheExpire_s;
  fUrlRe = new TPMERegexp("^alien://(.*)$");
  fUrlTpl = urlTpl;
  fUrlTpl.ReplaceAll("<path>", "$1");

  fCache = new TDataSetManagerFile("_cache_", "_cache_",
    Form("dir:%s perms:open", cacheDir.Data()));

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
  : TDataSetManager("", "", ""), fUrlRe(0), fCache(0)
{
  Init(cacheDir, urlTpl, cacheExpire_s);
}

//______________________________________________________________________________
TDataSetManagerAliEn::TDataSetManagerAliEn(const char *, const char *,
  const char *cfgStr) : TDataSetManager("", "", ""), fUrlRe(0), fCache(0)
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
}

//______________________________________________________________________________
TList *TDataSetManagerAliEn::GetFindCommandsFromUri(TString &uri,
  EDataMode &dataMode)
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
    TString treeName;
    TString regexp;

    // Custom search URI
    if (!ParseCustomFindUri(uri, basePath, fileName, anchor,
      treeName, regexp)) {
      Error("GetFindCommandsFromUri", "Malformed AliEn find command");
      return NULL;
    }

    findCommands = new TList();
    findCommands->SetOwner();
    findCommands->Add( new TAliEnFind(basePath, fileName, anchor, kFALSE,
      treeName, regexp) );

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

    for (UInt_t i=0; i<runList->size(); i++) {

      // Here we need to assemble the find string
      TString basePath, fileName, temp;

      if (sim) {

        //
        // Montecarlo
        //

        // Check whether this period is in /alice/sim/<period> or in
        // /alice/sim/<year>/<period> and act properly, since naming convention
        // is unclear!
        if (!gGrid) {
          TGrid::Connect("alien:");
          if (!gGrid) {
            delete findCommands;
            return NULL;
          }
        }

        basePath = Form("/alice/sim/%s", lhcPeriod.Data());  // no year
        if (!gGrid->Cd(basePath.Data())) {
          basePath = Form("/alice/sim/%d/%s", year, lhcPeriod.Data());
        }
        temp.Form("/%06d", runList->at(i));
        basePath.Append(temp);

        if (!esd) {
          temp = Form("/AOD%03d", aodNum);
          basePath.Append(temp);
        }
      }
      else {

        //
        // Real data
        //

        // Parse the pass string: if it starts with a number, prepend "pass"
        if ((pass[0] >= '0') && (pass[0] <= '9')) pass.Prepend("pass");

        // Data
        basePath = Form("/alice/data/%d/%s/%09d/ESDs/%s", year,
          lhcPeriod.Data(), runList->at(i), pass.Data());
        if (esd) {
          basePath.Append("/*.*");
        }
        else {
          temp = Form("/AOD%03d", aodNum);
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

  // If no valid data found, then findCommands is NULL
  return findCommands;
}

//______________________________________________________________________________
Bool_t TDataSetManagerAliEn::ParseCustomFindUri(TString &uri,
   TString &basePath, TString &fileName, TString &anchor, TString &treeName,
   TString &regexp)
{

  // Base path
  TPMERegexp reBasePath("(^|;)BasePath=([^; ]+)(;|$)");
  if (reBasePath.Match(uri) != 4) {
    Error("ParseCustomFindUri", "Base path not specified");
    return kFALSE;
  }
  basePath = reBasePath[2];

  // File name
  TPMERegexp reFileName("(^|;)FileName=([^; ]+)(;|$)");
  if (reFileName.Match(uri) != 4) {
    Error("ParseCustomFindUri", "File name not specified");
    return kFALSE;
  }
  fileName = reFileName[2];

  // Anchor (optional)
  TPMERegexp reAnchor("(^|;)Anchor=([^; ]+)(;|$)");
  if (reAnchor.Match(uri) != 4)
    anchor = "";
  else
    anchor = reAnchor[2];

  // Tree name (optional)
  TPMERegexp reTreeName("(^|;)Tree=(/[^; ]+)(;|$)");
  if (reTreeName.Match(uri) != 4)
    treeName = "";
  else
    treeName = reTreeName[2];

  // Regexp (optional)
  TPMERegexp reRegexp("(^|;)Regexp=([^; ]+)(;|$)");
  if (reRegexp.Match(uri) != 4)
    regexp = "";
  else
    regexp = reRegexp[2];

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TDataSetManagerAliEn::ParseOfficialDataUri(TString &uri, Bool_t sim,
  TString &period, Int_t &year, std::vector<Int_t> *&runList, Bool_t &esd,
  Int_t &aodNum, TString &pass)
{

  //
  // Parse LHC period
  //

  TPMERegexp rePeriod("(^|;)Period=(LHC([0-9]{2})[^;]*)(;|$)");
  if (rePeriod.Match(uri) != 5) {
    Error("ParseOfficialDataUri",
      "LHC period not specified (e.g. Period=LHC10h)");
    return kFALSE;
  }

  period = rePeriod[2];
  year = rePeriod[3].Atoi() + 2000;

  //
  // Parse data format (ESDs or AODXXX)
  //

  TPMERegexp reFormat("(^|;)Variant=(ESDs?|AOD([0-9]{3}))(;|$)");
  if (reFormat.Match(uri) != 5) {
    Error("ParseOfficialDataUri",
      "Data variant (e.g., Variant=ESD or AOD079) not specified");
    return kFALSE;
  }

  if (reFormat[2].BeginsWith("ESD")) esd = kTRUE;
  else {
    esd = kFALSE;
    aodNum = reFormat[3].Atoi();
  }

  //
  // Parse pass: mandatory on Data, useless on Sim
  //

  TPMERegexp rePass("(^|;)Pass=([a-zA-Z_0-9-]+)(;|$)");
  if ((rePass.Match(uri) != 4) && (!sim)) {
    Error("ParseOfficialDataUri",
      "Pass (e.g., Pass=cpass1_muon) is mandatory on real data");
    return kFALSE;
  }
  pass = rePass[2];

  //
  // Parse run list
  //

  TPMERegexp reRun("(^|;)Run=([0-9,-]+)(;|$)");
  if (reRun.Match(uri) != 4) {
    Error("ParseOfficialDataUri",
      "Run or run range not specified (e.g., Run=139104-139107,139306)");
    return kFALSE;
  }
  TString runListStr = reRun[2];
  runList = ExpandRunSpec(runListStr);  // must be freed by caller

  return kTRUE;
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

  EDataMode dataMode;
  TString sUri(uri);
  TList *findCmds = GetFindCommandsFromUri(sUri, dataMode);
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

    if ((mtime > 0) && (now-mtime > fCacheExpire_s)) {
      if (gDebug >= 1)
        Info("GetDataSet", "Dataset cache expired");
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
        tUrl = fi->GetCurrentUrl()->GetUrl();
        fUrlRe->Substitute(tUrl, fUrlTpl);
        fi->AddUrl(tUrl.Data(), kTRUE);  // kTRUE == prepend URL
        fi->ResetUrl();
      }

      // Add endpoint?
      if (dataMode == kDataLocal) {
        fillLocality = kTRUE;  // will fill
      }
      else {
        // Don't make the user waste time: don't cache dataset locality info at
        // this time, and signal our ignorance with a dummy URL
        if (gDebug >= 1)
          Info("GetDataSet", "Not caching data locality information now");
        itCache.Reset();
        while ((fi = dynamic_cast<TFileInfo *>(itCache.Next())))
          fi->AddUrl("noop://unknown", kTRUE);
      }

      // Update summary information and save to cache!
      saveToCache = kTRUE;

    } // end dataset just got from AliEn
    else {

      // Reading dataset from cache. Check if it has endpoint information.
      Bool_t hasEndp = kTRUE;

      fi = dynamic_cast<TFileInfo *>(newFc->GetList()->At(0));
      if (fi) {
        if ((strcmp(fi->GetCurrentUrl()->GetProtocol(), "noop") == 0) &&
            (strcmp(fi->GetCurrentUrl()->GetHost(), "unknown") == 0)) {
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
          fi->RemoveUrlAt(0);
          //fi->RemoveUrl("noop://unknown");
        }

        fillLocality = kTRUE;  // will locate
        saveToCache = kTRUE;  // will cache
      }

    } // end processing dataset from cache

    // Fill locality: initialize stager, locate URLs
    if (fillLocality) {
      fi = dynamic_cast<TFileInfo *>(newFc->GetList()->At(0));
      if (fi) {
        Info("GetDataSet", "Filling dataset locality information: "
          "it might take time, be patient!");

        // Lazy stager initialization
        if (!fstg) fstg = TFileStager::Open(fi->GetCurrentUrl()->GetUrl());

        if (!fstg) {  // seems nonsense but look carefully :)
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
    //newFc->Print("filter:SsCc");

    // Now we prepare the final dataset, by appending proper information from
    // newFc to fc

    TIter itCache(newFc->GetList());
    while ((fi = dynamic_cast<TFileInfo *>(itCache.Next()))) {

      // We no longer have unknowns. Instead we might have: redir, none. Let's
      // eliminate them. For each entry we always have three URLs

      if (dataMode == kDataRemote) {
        // Set everything as staged and remove first two URLs: only AliEn needed
        fi->SetBit(TFileInfo::kStaged);
        fi->RemoveUrlAt(0);
        fi->RemoveUrlAt(0);
      }
      else if (dataMode == kDataCache) {
        // Access from redirector, pretend that everything is staged
        fi->SetBit(TFileInfo::kStaged);
        fi->RemoveUrlAt(0);
      }
      else {  // dataMode == kLocal
        // Remove dummy URLs, trust staged bit
        fi->RemoveUrl("noop://none");
        fi->RemoveUrl("noop://redir");
      }

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
