// @(#)root/proof:$Id$
// Author: Dario Berzano, 26.11.12

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataSetManagerAliEn
#define ROOT_TDataSetManagerAliEn

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSetManagerAliEn                                                 //
//                                                                      //
// Implementation of TDataSetManager dynamically creating datasets      //
// by querying the AliEn file catalog.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDataSetManager
#include "TDataSetManager.h"
#endif

#ifndef ROOT_TDataSetManagerFile
#include "TDataSetManagerFile.h"
#endif

#include "TFileStager.h"
#include "TPRegexp.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TGridResult.h"
#include "TGrid.h"
#include "THashList.h"
#include "TSystem.h"

typedef enum { kDataRemote, kDataCache, kDataLocal } EDataMode;

class TAliEnFind : public TObject {

   private:

      TString      fBasePath;
      TString      fFileName;
      TString      fTreeName;
      TString      fRegexpRaw;
      TString      fAnchor;
      Bool_t       fArchSubst;
      TPMERegexp  *fRegexp;
      TString      fSearchId;
      TGridResult *fGridResult;

      inline virtual void InvalidateSearchId();
      inline virtual void InvalidateGridResult();

   public:

      TAliEnFind(const TString &basePath = "", const TString &fileName = "",
         const TString &anchor = "", const Bool_t archSubst = kFALSE,
         const TString &treeName = "", const TString &regexp = "");

      TAliEnFind(const TAliEnFind &src);
      TAliEnFind &operator=(const TAliEnFind &rhs);

      virtual                  ~TAliEnFind();

      virtual TGridResult      *GetGridResult(Bool_t forceNewQuery = kFALSE);

      virtual const TString    &GetBasePath()  const { return fBasePath; };
      virtual const TString    &GetFileName()  const { return fFileName; };
      virtual const TString    &GetAnchor()    const { return fAnchor; };
      virtual const TString    &GetTreeName()  const { return fTreeName; };
      virtual       Bool_t      GetArchSubst() const { return fArchSubst; };
      virtual const TPMERegexp *GetRegexp()    const { return fRegexp; };

      virtual void              SetBasePath(const char *basePath);
      virtual void              SetFileName(const char *fileName);
      virtual void              SetAnchor(const char *anchor);
      virtual void              SetTreeName(const char *fileName);
      virtual void              SetArchSubst(Bool_t archSubst);
      virtual void              SetRegexp(const char *regexp);

      virtual const char       *GetSearchId();
      virtual TFileCollection  *GetCollection(Bool_t forceNewQuery = kFALSE);
      virtual void              Print(Option_t* opt = "") const;

  ClassDef(TAliEnFind, 0);  // Interface to the AliEn find command

};

class TDataSetManagerAliEn : public TDataSetManager {

   protected:

      TPMERegexp          *fUrlRe;
      TString              fUrlTpl;
      TDataSetManagerFile *fCache;
      Long_t               fCacheExpire_s;

      std::vector<Int_t> *ExpandRunSpec(TString &runSpec);

      virtual Bool_t ParseCustomFindUri(TString &uri, TString &basePath,
         TString &fileName, TString &anchor, TString &treeName,
         TString &regexp);

      virtual Bool_t ParseOfficialDataUri(TString &uri, Bool_t sim,
         TString &period, Int_t &year, std::vector<Int_t> *&runList,
         Bool_t &esd, Int_t &aodNum, TString &pass);

      virtual void Init(TString cacheDir, TString urlTpl,
         ULong_t cacheExpire_s);

   public:

      TDataSetManagerAliEn() : TDataSetManager(0, 0, 0) {}
      TDataSetManagerAliEn(const char *cacheDir, const char *urlTpl,
         ULong_t cacheExpire_s);
      TDataSetManagerAliEn(const char *, const char *, const char *cfgStr);

      virtual TList *GetFindCommandsFromUri(TString &uri, EDataMode &dataMode);

      virtual ~TDataSetManagerAliEn();
      virtual TFileCollection *GetDataSet(const char *uri, const char * = 0);
      virtual Bool_t ExistsDataSet(const char *uri);

      // Not implemented functionalities
      virtual Int_t RegisterDataSet(const char *, TFileCollection *,
         const char *);
      virtual TMap *GetDataSets(const char *, UInt_t);
      virtual void  ShowDataSets(const char * = "*", const char * = "");
      virtual Bool_t RemoveDataSet(const char *uri);
      virtual Int_t ScanDataSet(const char *, UInt_t);
      virtual Int_t ShowCache(const char *);
      virtual Int_t ClearCache(const char *);

   ClassDef(TDataSetManagerAliEn, 0) // Dataset to AliEn catalog interface
};

#endif
