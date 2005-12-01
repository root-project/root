// @(#)root/net:$Name:  $:$Id: TKeySQL.cxx,v 1.3 2005/11/28 23:22:31 pcanal Exp $
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
// TKeySQL is represents a metainforamtion about object, which was written to
// SQL database. It keeps object id, which used to locate object data
// from database tables.
//________________________________________________________________________


#include "TKeySQL.h"

#include "TClass.h"
#include "TBrowser.h"
#include "Riostream.h"

#include "TSQLResult.h"
#include "TBufferSQL2.h"
#include "TSQLStructure.h"
#include "TSQLFile.h"

ClassImp(TKeySQL);

//______________________________________________________________________________
TKeySQL::TKeySQL() :
   TKey(),
   fFile(0),
   fKeyId(-1)
{
   // default constructor
}

//______________________________________________________________________________
TKeySQL::TKeySQL(TSQLFile* file, const TObject* obj, const char* name) :
    TKey(),
    fFile(file),
    fKeyId(-1),
    fObjId(-1)
{
   // Creates TKeySQL and convert obj data to TSQLStructure via TBufferSQL2

   if (name) SetName(name); else
      if (obj!=0) {SetName(obj->GetName());  fClassName=obj->ClassName();}
      else SetName("Noname");

   StoreObject((void*)obj, obj ? obj->IsA() : 0);
}

//______________________________________________________________________________
TKeySQL::TKeySQL(TSQLFile* file, const void* obj, const TClass* cl, const char* name) :
    TKey(),
    fFile(file),
    fKeyId(-1),
    fObjId(-1)
{
   // Creates TKeySQL and convert obj data to TSQLStructure via TBufferSQL2

   if (name && *name) SetName(name);
   else SetName(cl ? cl->GetName() : "Noname");

   StoreObject(obj, cl);
}

//______________________________________________________________________________
TKeySQL::TKeySQL(TSQLFile* file, Int_t keyid, Int_t dirid, Int_t objid, const char* name,
                 const char* keydatetime, Int_t cycle, const char* classname) :
    TKey(),
    fFile(file),
    fKeyId(keyid),
    fDirId(dirid),
    fObjId(objid)
{
   // Create TKeySQL object, which correponds to single entry in keys table

   SetName(name);
   TDatime dt(keydatetime);
   fDatime = dt;
   fCycle = cycle;
   fClassName = classname;
}

//______________________________________________________________________________
TKeySQL::~TKeySQL()
{
// TKeySQL destructor
}

//______________________________________________________________________________
void TKeySQL::Browse(TBrowser *b)
{
// Browse object corresponding to this key

   TObject *obj = gDirectory->GetList()->FindObject(GetName());
   if (obj && !obj->IsFolder()) {
      if (obj->InheritsFrom(TCollection::Class()))
         obj->Delete();   // delete also collection elements
      delete obj;
      obj = 0;
   }

   if (!obj)
      obj = ReadObj();

   if (b && obj) {
      obj->Browse(b);
      b->SetRefreshFlag(kTRUE);
   }
}

//______________________________________________________________________________
void TKeySQL::Delete(Option_t * /*option*/)
{
// Removes key from current directory
// Note: TKeySQL object is not deleted. You still have to call "delete key"

   if (fFile!=0)
      fFile->DeleteKeyFromDB(GetDBKeyId());

   gDirectory->GetListOfKeys()->Remove(this);
}

//______________________________________________________________________________
void TKeySQL::StoreObject(const void* obj, const TClass* cl)
{
//  convert object to sql statements and store them in DB

   TObjArray cmds;
   Bool_t needcommit = kFALSE;

   fCycle = fFile->AppendKey(this);

   fKeyId = fFile->DefineNextKeyId();

   fObjId = fFile->VerifyObjectTable();
   if (fObjId<=0) fObjId = 1;
   else fObjId++;

   TBufferSQL2 buffer(TBuffer::kWrite, fFile);

   TSQLStructure* s = buffer.SqlWrite(obj, cl, fObjId);

   if ((buffer.GetErrorFlag()>0) || (s==0)) {
      Error("StoreObject","Cannot convert object data to TSQLStructure");
      goto zombie;
   }

   if (cl) fClassName = cl->GetName();
   
   if (fFile->GetUseTransactions()==TSQLFile::kTransactionsAuto) {
      fFile->SQLStartTransaction();
      needcommit = kTRUE;
   }

   // here tables may be already created, therefore 
   // it should be protected by transactions operations
   if (!s->ConvertToTables(fFile, fKeyId, &cmds)) {
      Error("StoreObject","Cannot convert to SQL statements");
      goto zombie;
   }
   
   if (!fFile->SQLApplyCommands(&cmds)) {
      Error("StoreObject","Cannot correctly store object data in database");
      goto zombie; 
   }

   // commit here, while objects data is already stored 
   if (needcommit) {
      fFile->SQLCommit();
      needcommit = kFALSE;
   }

   cmds.Delete();

   fDatime.Set();
   
   if (!fFile->WriteKeyData(GetDBKeyId(),
                            sqlio::Ids_RootDir, // later parent directory id should be
                            GetDBObjId(),
                            GetName(),
                            GetDatime().AsSQLString(),
                            GetCycle(),
                            GetClassName())) {
      // cannot add entry to keys table                          
      Error("StoreObject","Cannot write data to key tables");
      // delete everything relevant for that key
      fFile->DeleteKeyFromDB(GetDBKeyId());
      goto zombie;                           
   }

   return;
   
zombie:
   if (needcommit)
      fFile->SQLRollback();

   cmds.Delete();
   gDirectory->GetListOfKeys()->Remove(this);
   // fix me !!! One should delete object by other means
   // delete this;
}

//______________________________________________________________________________
TObject* TKeySQL::ReadObj()
{
// Read object derived from TObject class
// If it is not TObject or in case of error, return 0

   if (gDebug>0)
      cout << "TKeySQL::ReadObj fKeyId = " << fKeyId << endl;

   if ((fKeyId<=0) || (fFile==0)) return 0;

   TBufferSQL2 buffer(TBuffer::kRead, fFile);

   TObject* obj = buffer.SqlRead(fObjId);

   return obj;
}

//______________________________________________________________________________
void* TKeySQL::ReadObjectAny(const TClass* /*cl*/)
{
// read object of any type from SQL database

   if ((fKeyId<=0) || (fFile==0)) return 0;

   TBufferSQL2 buffer(TBuffer::kRead, fFile);

   void* obj = buffer.SqlReadAny(fObjId, 0);

   return obj;
}

