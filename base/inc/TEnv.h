// @(#)root/base:$Name:  $:$Id: TEnv.h,v 1.4 2000/12/13 15:13:45 brun Exp $
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnv                                                                 //
//                                                                      //
// The TEnv class reads a config file, by default .rootrc. Three types  //
// of .rootrc files are read: global, user and local files. The global  //
// file resides in $ROOTSYS, the user file in ~/ and the local file in  //
// the current working directory.                                       //
// The format of the .rootrc file is similar to the .Xdefaults format:  //
//                                                                      //
//   <SystemName>.<RootName|ProgName>.<name>[(type)]:  <value>          //
//                                                                      //
// Where <SystemName> is either, Unix, Mac or Dos (anything from MS),   //
// <RootName> the root name as given in the TROOT ctor,                 //
// <ProgName> the current program name and                              //
// <name> is the resource name, with optionally a type specification.   //
//                                                                      //
// E.g.:                                                                //
//                                                                      //
//   Unix.rint.Root.DynamicPath: .:$ROOTSYS/lib:~/lib                   //
//   Rint.Root.Debug:  FALSE                                            //
//   TH.Root.Debug: YES                                                 //
//   *.Root.MemStat: 1                                                  //
//                                                                      //
// <SystemName> and <ProgName> or <RootName> may be the wildcard "*".   //
// A # in the first column starts comment line.                         //
//                                                                      //
// Currently the following resources are defined:                       //
//    Root.Debug                (bool)         (*)                      //
//    Root.MemStat              (int)          (*)                      //
//    Root.MemStat.size         (int)          (*)                      //
//    Root.MemStat.cnt          (int)          (*)                      //
//    Root.MemCheck             (bool)         (*)                      //
//    Root.DynamicPath          (string)                                //
//    Rint.Logon                (string)                                //
//    Rint.Logoff               (string)                                //
//                                                                      //
// (*) work only with the <RootName> since no <ProgName> is available   //
//     at time of initialization.                                       //
//                                                                      //
// Note that the .rootrc config files contain the config for all ROOT   //
// based applications.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TObject.h"
#include "TString.h"

class TOrdCollection;
class TEnv;
class TEnvParser;
class TReadEnvParser;
class TWriteEnvParser;

enum EEnvLevel {
   kEnvGlobal,
   kEnvUser,
   kEnvLocal,
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
   TString     fName;
   TString     fType;
   TString     fValue;
   EEnvLevel   fLevel;
   Bool_t      fModified;
   TObject    *fObject;

   TEnvRec() { }

   TEnvRec(const char *n, const char *v, const char *t, EEnvLevel l);
   TEnvRec(const char *n, const TString &v, const char *t, EEnvLevel l);
   Int_t    Compare(const TObject *obj) const;
   void     ChangeValue(const char *v, const char *t, EEnvLevel l);
   void     ChangeValue(const TString &v, const char *t, EEnvLevel l);
   TString  ExpandValue(const char *v);
   void     Read(TObject *obj);
   Int_t    Read(const char *name) { return TObject::Read(name); }
   void     Write(TObject *obj);
   Int_t    Write(const char *name=0, Int_t opt=0, Int_t bufs=0)
                                     { return TObject::Write(name, opt, bufs); }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEnv                                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TEnv : public TObject {

private:
   TOrdCollection   *fTable;
   TString           fRcName;

   const char       *Getvalue(const char *name);

public:
   TEnv(const char *name="");
   virtual ~TEnv();

   TOrdCollection     *GetTable() const { return fTable; }
   Bool_t              Defined(const char *name)
                                  { return Getvalue(name) != 0; }

   virtual Int_t       GetValue(const char *name, Int_t dflt);
   virtual Double_t    GetValue(const char *name, Double_t dflt);
   virtual const char *GetValue(const char *name, const char *dflt);
   virtual TObject    *GetValue(const char *name, TObject *dflt);

   virtual void        SetValue(const char *name, const char *value,
                                EEnvLevel level = kEnvChange, const char *type = 0);
   virtual void        SetValue(const char *name, const TString &value,
                                EEnvLevel level, const char *type);
   virtual void        SetValue(const char *name, EEnvLevel level = kEnvChange);
   virtual void        SetValue(const char *name, Int_t value);
   virtual void        SetValue(const char *name, Double_t value);

   virtual TEnvRec    *Lookup(const char *n);
   virtual void        ReadFile(const char *fname, EEnvLevel level);
   virtual void        Save();
   virtual void        SaveLevel(EEnvLevel level);
   virtual void        Print(Option_t *option="") const;
   virtual void        PrintEnv(EEnvLevel level = kEnvAll) const;

   ClassDef(TEnv,0)  //Handle ROOT configuration resources
};

R__EXTERN TEnv *gEnv;

#endif
