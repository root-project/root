// @(#)root/base:$Name:$:$Id:$
// Author: Fons Rademakers   16/09/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualProof
#define ROOT_TVirtualProof


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualProof                                                        //
//                                                                      //
// Abstract interface to the Parallel ROOT Facility, PROOF.             //
// For more information on PROOF see the TProof class.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

typedef long Long64_t;

class TList;
class TDSet;
class TEventList;


class TVirtualProof : public TObject {

protected:
   TVirtualProof() { }

public:
   TVirtualProof(const char *masterurl, const char *conffile = 0,
                 const char *confdir = 0, Int_t loglevel = 1) { }
   virtual ~TVirtualProof() { }

   virtual Int_t       Ping() = 0;
   virtual Int_t       Exec(const char *cmd) = 0;
   virtual Int_t       Process(TDSet *set, const char *selector,
                       Long64_t nentries = -1, Long64_t first = 0,
                       TEventList *evl = 0) = 0;

   virtual void        AddInput(TObject *obj) = 0;
   virtual void        ClearInput() = 0;
   virtual TObject    *GetOutput(const char *name) = 0;
   virtual TList      *GetOutputList() = 0;

   virtual Int_t       SetParallel(Int_t nodes = 9999) = 0;
   virtual void        SetLogLevel(Int_t level) = 0;

   virtual void        Close(Option_t *option="") = 0;
   virtual void        Print(Option_t *option="") const = 0;

   virtual void        ShowCache(Bool_t all = kFALSE) = 0;
   virtual void        ClearCache() = 0;
   virtual void        ShowPackages(Bool_t all = kFALSE) = 0;
   virtual void        ShowEnabledPackages(Bool_t all = kFALSE) = 0;
   virtual void        ClearPackages() = 0;
   virtual void        ClearPackage(const char *package) = 0;
   virtual Int_t       EnablePackage(const char *package) = 0;
   virtual Int_t       UploadPackage(const char *par, Int_t parallel = 1) = 0;

   virtual const char *GetMaster() const = 0;
   virtual const char *GetConfDir() const = 0;
   virtual const char *GetConfFile() const = 0;
   virtual const char *GetUser() const = 0;
   virtual const char *GetWorkDir() const = 0;
   virtual const char *GetImage() const = 0;
   virtual Int_t       GetPort() const = 0;
   virtual Int_t       GetProtocol() const = 0;
   virtual Int_t       GetStatus() const = 0;
   virtual Int_t       GetLogLevel() const = 0;
   virtual Int_t       GetParallel() const = 0;

   virtual Double_t    GetBytesRead() const = 0;
   virtual Float_t     GetRealTime() const = 0;
   virtual Float_t     GetCpuTime() const = 0;

   virtual Bool_t      IsMaster() const = 0;
   virtual Bool_t      IsValid() const = 0;
   virtual Bool_t      IsParallel() const = 0;

   ClassDef(TVirtualProof,0)  // Abstract PROOF interface
};

R__EXTERN TVirtualProof *gProof;

#endif
