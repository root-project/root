// $Id$
// Author: Sergey Linev   22/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootSnifferStore
#define ROOT_TRootSnifferStore

#include "TObject.h"

#include "TString.h"

class TDataMember;
class TFolder;

/** Abstract interface for storage of hierarchy scan in TRootSniffer */

class TRootSnifferStore : public TObject {
protected:
   void *fResPtr{nullptr};           ///<! pointer on found item
   TClass *fResClass{nullptr};       ///<! class of found item
   TDataMember *fResMember{nullptr}; ///<! datamember pointer of found item
   Int_t fResNumChilds{-1};          ///<! count of found childs, -1 by default
   Int_t fResRestrict{0};            ///<! restriction for result, 0-default, 1-readonly, 2-full
public:
   virtual ~TRootSnifferStore() = default;

   virtual void CreateNode(Int_t, const char *) {}
   virtual void SetField(Int_t, const char *, const char *, Bool_t) {}
   virtual void BeforeNextChild(Int_t, Int_t, Int_t) {}
   virtual void CloseNode(Int_t, Int_t) {}

   void SetResult(void *_res, TClass *_rescl, TDataMember *_resmemb, Int_t _res_chld, Int_t restr = 0);

   void *GetResPtr() const { return fResPtr; }
   TClass *GetResClass() const { return fResClass; }
   TDataMember *GetResMember() const { return fResMember; }
   Int_t GetResNumChilds() const { return fResNumChilds; }
   Int_t GetResRestrict() const { return fResRestrict; }
   virtual Bool_t IsXml() const { return kFALSE; }

   ClassDefOverride(TRootSnifferStore, 0) // structure for results store of objects sniffer
};

// ========================================================================

/** Storage of hierarchy scan in TRootSniffer in XML format */

class TRootSnifferStoreXml : public TRootSnifferStore {
protected:
   TString &fBuf;           ///<! output buffer
   Bool_t fCompact{kFALSE}; ///<! produce compact xml code

public:
   explicit TRootSnifferStoreXml(TString &_buf, Bool_t _compact = kFALSE) : TRootSnifferStore(), fBuf(_buf), fCompact(_compact)
   {
   }

   void CreateNode(Int_t lvl, const char *nodename) final;
   void SetField(Int_t lvl, const char *field, const char *value, Bool_t) final;
   void BeforeNextChild(Int_t lvl, Int_t nchld, Int_t) final;
   void CloseNode(Int_t lvl, Int_t numchilds) final;

   Bool_t IsXml() const final { return kTRUE; }

   ClassDefOverride(TRootSnifferStoreXml, 0) // xml results store of objects sniffer
};

// ========================================================================

/** Storage of hierarchy scan in TRootSniffer in JSON format */

class TRootSnifferStoreJson : public TRootSnifferStore {
protected:
   TString &fBuf;           ///<! output buffer
   Bool_t fCompact{kFALSE}; ///<! produce compact json code
public:
   explicit TRootSnifferStoreJson(TString &_buf, Bool_t _compact = kFALSE) : TRootSnifferStore(), fBuf(_buf), fCompact(_compact)
   {
   }

   void CreateNode(Int_t lvl, const char *nodename) final;
   void SetField(Int_t lvl, const char *field, const char *value, Bool_t with_quotes) final;
   void BeforeNextChild(Int_t lvl, Int_t nchld, Int_t nfld) final;
   void CloseNode(Int_t lvl, Int_t numchilds) final;

   ClassDefOverride(TRootSnifferStoreJson, 0) // json results store of objects sniffer
};

#endif
