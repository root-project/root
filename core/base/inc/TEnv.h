// @(#)root/base:$Id$
// Author: Fons Rademakers   22/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEnv
#define ROOT_TEnv

#include "TObject.h"
#include "TString.h"

class THashList;
class TEnv;
class TEnvParser;
class TReadEnvParser;
class TWriteEnvParser;

enum EEnvLevel {
   kEnvGlobal,
   kEnvUser,
   kEnvLocal, // For gEnv, the local level is disabled by default
   kEnvChange,
   kEnvAll
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnvRec                                                              //
//                                                                      //
// Individual TEnv records.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TEnvRec : public TObject {

friend class  TEnv;
friend class  TEnvParser;
friend class  TReadEnvParser;
friend class  TWriteEnvParser;

private:
   TString     fName;       // env rec key name
   TString     fType;       // env rec type
   TString     fValue;      // env rec value
   EEnvLevel   fLevel;      // env rec level
   Bool_t      fModified;   // if env rec has been modified

   TEnvRec(const char *n, const char *v, const char *t, EEnvLevel l);
   Int_t    Compare(const TObject *obj) const override;
   void     ChangeValue(const char *v, const char *t, EEnvLevel l,
                        Bool_t append = kFALSE, Bool_t ignoredup = kFALSE);
   TString  ExpandValue(const char *v);

public:
   TEnvRec(): fName(), fType(), fValue(), fLevel(kEnvAll), fModified(kTRUE) { }
   ~TEnvRec();
   const char *GetName() const override { return fName; }
   const char *GetValue() const { return fValue; }
   const char *GetType() const { return fType; }
   EEnvLevel   GetLevel() const { return fLevel; }
   ULong_t     Hash() const override { return fName.Hash(); }

   ClassDefOverride(TEnvRec,2)  // Individual TEnv records
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnv                                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TEnv : public TObject {

private:
   THashList        *fTable;                // hash table containing env records
   TString           fRcName;               // resource file base name
   Bool_t            fIgnoreDup;            // ignore duplicates, don't issue warning
   Bool_t            fIsLocalLevelDisabled; //! By default, gEnv does not allow use of the local level

   TEnv(const TEnv&) = delete;
   TEnv& operator=(const TEnv&) = delete;

   const char       *Getvalue(const char *name) const;
   const char       *GetUserDirectory() const;

public:
   TEnv(const char *name = "", bool disableLocalLevel = false);
   virtual ~TEnv();

   THashList          *GetTable() const { return fTable; }
   Bool_t              Defined(const char *name) const
                                    { return Getvalue(name) != nullptr; }

   virtual const char *GetRcName() const { return fRcName; }
   virtual void        SetRcName(const char *name) { fRcName = name; }

   virtual Int_t       GetValue(const char *name, Int_t dflt) const;
   virtual Double_t    GetValue(const char *name, Double_t dflt) const;
   virtual const char *GetValue(const char *name, const char *dflt) const;

   virtual void        SetValue(const char *name, const char *value,
                                EEnvLevel level = kEnvChange,
                                const char *type = nullptr);
   virtual void        SetValue(const char *name, EEnvLevel level = kEnvChange);
   virtual void        SetValue(const char *name, Int_t value);
   virtual void        SetValue(const char *name, Double_t value);

   virtual TEnvRec    *Lookup(const char *n) const;
   virtual Int_t       ReadFile(const char *fname, EEnvLevel level);
   virtual Int_t       WriteFile(const char *fname, EEnvLevel level = kEnvAll);
   virtual void        Save();
   virtual void        SaveLevel(EEnvLevel level);
   void                Print(Option_t *option="") const override;
   virtual void        PrintEnv(EEnvLevel level = kEnvAll) const;
   Bool_t              IgnoreDuplicates(Bool_t ignore);

   Bool_t              IsLocalLevelDisabled() const { return fIsLocalLevelDisabled; }

   ClassDefOverride(TEnv,2)  // Handle ROOT configuration resources
};

R__EXTERN TEnv *gEnv;

#endif
