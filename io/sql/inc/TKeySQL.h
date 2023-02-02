// @(#)root/sql:$Id$
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TKeySQL
#define ROOT_TKeySQL

#include "TKey.h"

class TSQLFile;

class TKeySQL final : public TKey {

private:
   TKeySQL(const TKeySQL &) = delete;
   TKeySQL &operator=(const TKeySQL &) = delete;

protected:
   TKeySQL() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see ROOT-10300

   using TKey::Read;

   void StoreKeyObject(const void *obj, const TClass *cl);
   void *ReadKeyObject(void *obj, const TClass *expectedClass);

   Long64_t fKeyId{-1}; ///<!  key identifier in KeysTables
   Long64_t fObjId{-1}; ///<!  stored object identifier

public:
   TKeySQL(TDirectory *mother, const TObject *obj, const char *name, const char *title = nullptr);
   TKeySQL(TDirectory *mother, const void *obj, const TClass *cl, const char *name, const char *title = nullptr);
   TKeySQL(TDirectory *mother, Long64_t keyid, Long64_t objid, const char *name, const char *title,
           const char *keydatetime, Int_t cycle, const char *classname);
   virtual ~TKeySQL() = default;

   Bool_t IsKeyModified(const char *keyname, const char *keytitle, const char *keydatime, Int_t cycle, const char *classname);

   Long64_t GetDBKeyId() const { return fKeyId; }
   Long64_t GetDBObjId() const { return fObjId; }
   Long64_t GetDBDirId() const;

   // redefined TKey Methods
   void Delete(Option_t *option = "") final;
   void DeleteBuffer() final {}
   void FillBuffer(char *&) final {}
   char *GetBuffer() const final { return nullptr; }
   Long64_t GetSeekKey() const final { return GetDBObjId() > 0 ? GetDBObjId() : 0; }
   Long64_t GetSeekPdir() const final { return GetDBDirId() > 0 ? GetDBDirId() : 0; }
   void Keep() final {}

   Int_t Read(TObject *obj) final;
   TObject *ReadObj() final;
   TObject *ReadObjWithBuffer(char *bufferRead) final;
   void *ReadObjectAny(const TClass *expectedClass) final;

   void ReadBuffer(char *&) final {}
   Bool_t ReadFile() final { return kTRUE; }
   void SetBuffer() final { fBuffer =  nullptr; }
   Int_t WriteFile(Int_t = 1, TFile * = nullptr) final { return 0; }

   ClassDefOverride(TKeySQL, 1) // a special TKey for SQL data base
};

#endif
