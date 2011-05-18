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


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TKeySQL is TKey class from TSQLFile                                 //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TKey
#include "TKey.h"
#endif

class TSQLFile;

class TKeySQL : public TKey {
    
private:
   TKeySQL(const TKeySQL&);            // TKeySQL objects are not copiable.
   TKeySQL& operator=(const TKeySQL&); // TKeySQL objects are not copiable.
   
protected:
   TKeySQL();

   virtual Int_t     Read(const char *name) { return TKey::Read(name); }
   void              StoreKeyObject(const void* obj, const TClass* cl);
   void*             ReadKeyObject(void* obj, const TClass* expectedClass);
  
   Long64_t          fKeyId;    //!  key identifier in KeysTables
   Long64_t          fObjId;    //!  stored object identifer

public:
   TKeySQL(TDirectory* mother, const TObject* obj, const char* name, const char* title = 0);
   TKeySQL(TDirectory* mother, const void* obj, const TClass* cl, const char* name, const char* title = 0);
   TKeySQL(TDirectory* mother, Long64_t keyid, Long64_t objid, 
           const char* name, const char* title,
           const char* keydatetime, Int_t cycle, const char* classname);
   virtual ~TKeySQL();

   Bool_t            IsKeyModified(const char* keyname, const char* keytitle, const char* keydatime, Int_t cycle, const char* classname);
  
   Long64_t          GetDBKeyId() const { return fKeyId; }
   Long64_t          GetDBObjId() const { return fObjId; }
   Long64_t          GetDBDirId() const;

   // redefined TKey Methods
   virtual void      Delete(Option_t *option="");
   virtual void      DeleteBuffer() {}
   virtual void      FillBuffer(char *&) {}
   virtual char     *GetBuffer() const { return 0; }
   virtual Long64_t  GetSeekKey() const  { return GetDBObjId() > 0 ? GetDBObjId() : 0; }
   virtual Long64_t  GetSeekPdir() const { return GetDBDirId() > 0 ? GetDBDirId() : 0; }
   virtual void      Keep() {}

   virtual Int_t     Read(TObject* obj);
   virtual TObject  *ReadObj();
   virtual TObject  *ReadObjWithBuffer(char *bufferRead);
   virtual void     *ReadObjectAny(const TClass *expectedClass);
  
   virtual void      ReadBuffer(char *&) {}
   virtual Bool_t    ReadFile() { return kTRUE; }
   virtual void      SetBuffer() { fBuffer = 0; }
   virtual Int_t     WriteFile(Int_t =1, TFile* = 0) { return 0; }

   ClassDef(TKeySQL,1) // a special TKey for SQL data base
};

#endif
