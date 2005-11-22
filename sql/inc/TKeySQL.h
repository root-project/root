// @(#)root/net:$Name:  $:$Id: TKeySQL.h,v 1.2 2005/11/22 11:30:00 brun Exp $
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
    
protected:
   TKeySQL();

   virtual Int_t     Read(const char *name) { return TKey::Read(name); }
   void              StoreObject(const void* obj, const TClass* cl);
  
   TSQLFile*         fFile;     //!  pointer on SQL file
   Int_t             fKeyId;    //!  key identifier in KeysTables
   Int_t             fDirId;    //!  parent directory identifier
   Int_t             fObjId;    //!  stored object identifer

public:
   TKeySQL(TSQLFile* file, const TObject* obj, const char* name);
   TKeySQL(TSQLFile* file, const void* obj, const TClass* cl, const char* name);
   TKeySQL(TSQLFile* file, Int_t keyid, Int_t dirid, Int_t objid, const char* name, 
           const char* keydatetime, Int_t cycle, const char* classname);
   virtual ~TKeySQL();
  
   Int_t             GetDBKeyId() const { return fKeyId; }
   Int_t             GetDBDirId() const { return fDirId; }
   Int_t             GetDBObjId() const { return fObjId; }

   // redefined TKey Methods
   virtual void      Browse(TBrowser *b);
   virtual void      Delete(Option_t *option="");
   virtual void      DeleteBuffer() {}
   virtual void      FillBuffer(char *&) {}
   virtual char     *GetBuffer() const { return 0; }
   virtual Long64_t  GetSeekKey() const  { return 1; }
   virtual Long64_t  GetSeekPdir() const { return 1;}
   virtual void      Keep() {}

   virtual Int_t     Read(TObject*) { return 0; }
   virtual TObject  *ReadObj();
   virtual void     *ReadObjectAny(const TClass *cl);
  
   virtual void      ReadBuffer(char *&) {}
   virtual void      ReadFile() {}
   virtual void      SetBuffer() { fBuffer = 0; }
   virtual void      SetParent(const TObject* ) { }
   virtual Int_t     Sizeof() const { return 0; }
   virtual Int_t     WriteFile(Int_t =1) { return 0; }

   ClassDef(TKeySQL,1) // a special TKey for SQL data base
};

#endif
