// $Id$
// Author: Sergey Linev   22/12/2013

#ifndef ROOT_TRootSnifferStore
#define ROOT_TRootSnifferStore

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

class TDataMember;

class TRootSnifferStore : public TObject {
protected:
   void *fResPtr;           //! pointer on found item
   TClass *fResClass;       //! class of found item
   TDataMember *fResMember; //! datamember pointer of found item
   Int_t fResNumChilds;     //! count of found childs, -1 by default
public:
   TRootSnifferStore();
   virtual ~TRootSnifferStore();

   virtual void CreateNode(Int_t, const char *) {}
   virtual void SetField(Int_t, const char *, const char *, Bool_t) {}
   virtual void BeforeNextChild(Int_t, Int_t, Int_t) {}
   virtual void CloseNode(Int_t, const char *, Int_t) {}

   void SetResult(void *_res, TClass *_rescl, TDataMember *_resmemb, Int_t _res_chld);

   void *GetResPtr() const
   {
      return fResPtr;
   }
   TClass *GetResClass() const
   {
      return fResClass;
   }
   TDataMember *GetResMember() const
   {
      return fResMember;
   }
   Int_t GetResNumChilds() const
   {
      return fResNumChilds;
   }

   virtual Bool_t IsXml() const { return kFALSE; }

   ClassDef(TRootSnifferStore, 0) // structure for results store of objects sniffer
};

// ========================================================================

class TRootSnifferStoreXml : public TRootSnifferStore {
protected:
   TString *buf;           //! output buffer
   Bool_t compact;         //! produce compact xml code

public:
   TRootSnifferStoreXml(TString &_buf, Bool_t _compact = kFALSE) :
      TRootSnifferStore(),
      buf(&_buf),
      compact(_compact){}

   virtual ~TRootSnifferStoreXml() {}

   virtual void CreateNode(Int_t lvl, const char *nodename);
   virtual void SetField(Int_t lvl, const char *field, const char *value, Bool_t);
   virtual void BeforeNextChild(Int_t lvl, Int_t nchld, Int_t);
   virtual void CloseNode(Int_t lvl, const char *nodename, Int_t numchilds);

   virtual Bool_t IsXml() const { return kTRUE; }

   ClassDef(TRootSnifferStoreXml, 0) // xml results store of objects sniffer
};

// ========================================================================


class TRootSnifferStoreJson : public TRootSnifferStore {
protected:
   TString *buf;     //! output buffer
   Bool_t compact;   //! produce compact json code
public:
   TRootSnifferStoreJson(TString &_buf, Bool_t _compact = kFALSE) :
      TRootSnifferStore(),
      buf(&_buf),
      compact(_compact) {}
   virtual ~TRootSnifferStoreJson() {}

   virtual void CreateNode(Int_t lvl, const char *nodename);
   virtual void SetField(Int_t lvl, const char *field, const char *value, Bool_t with_quotes);
   virtual void BeforeNextChild(Int_t lvl, Int_t nchld, Int_t nfld);
   virtual void CloseNode(Int_t lvl, const char *nodename, Int_t numchilds);

   ClassDef(TRootSnifferStoreJson, 0) // json results store of objects sniffer
};

#endif
