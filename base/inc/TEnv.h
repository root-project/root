// @(#)root/base:$Name:  $:$Id: TEnv.h,v 1.5 2001/09/25 16:16:21 rdm Exp $
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
// file resides in $ROOTSYS/etc, the user file in ~/ and the local file //
// in the current working directory.                                    //
// The format of the .rootrc file is similar to the .Xdefaults format:  //
//                                                                      //
//   [+]<SystemName>.<RootName|ProgName>.<name>[(type)]:  <value>       //
//                                                                      //
// Where <SystemName> is either Unix, WinNT, MacOS or Vms,              //
// <RootName> the name as given in the TApplication ctor (or "RootApp"  //
// in case no explicit TApplication derived object was created),        //
// <ProgName> the current program name and <name> the resource name,    //
// with optionally a type specification. <value> can be either a        //
// string, an integer, a float/double or a boolean with the values      //
// TRUE, FALSE, ON, OFF, YES, NO, OK, NOT. Booleans will be returned as //
// an integer 0 or 1. The options [+] allows the concatenation of       //
// values to the same resouce name.                                     //
//                                                                      //
// E.g.:                                                                //
//                                                                      //
//   Unix.Rint.Root.DynamicPath: .:$ROOTSYS/lib:~/lib                   //
//   myapp.Root.Debug:  FALSE                                           //
//   TH.Root.Debug: YES                                                 //
//   *.Root.MemStat: 1                                                  //
//                                                                      //
// <SystemName> and <ProgName> or <RootName> may be the wildcard "*".   //
// A # in the first column starts comment line.                         //
//                                                                      //
// For the currently defined resources (and their default values) see   //
// $ROOTSYS/etc/system.rootrc.                                          //
//                                                                      //
// Note that the .rootrc config files contain the config for all ROOT   //
// based applications.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

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

   TEnvRec() { }

   TEnvRec(const char *n, const char *v, const char *t, EEnvLevel l);
   Int_t    Compare(const TObject *obj) const;
   void     ChangeValue(const char *v, const char *t, EEnvLevel l,
                        Bool_t append = kFALSE);
   TString  ExpandValue(const char *v);

public:
   const char *GetName() const { return fName; }
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

   virtual void        SetValue(const char *name, const char *value,
                                EEnvLevel level = kEnvChange,
                                const char *type = 0);
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
