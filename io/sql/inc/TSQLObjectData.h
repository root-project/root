// @(#)root/sql:$Id$
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLObjectData
#define ROOT_TSQLObjectData

#include "TObject.h"

#include "TString.h"

class TObjArray;
class TList;
class TSQLClassInfo;
class TSQLResult;
class TSQLRow;
class TSQLStatement;

class TSQLObjectInfo : public TObject {

public:
   TSQLObjectInfo();
   TSQLObjectInfo(Long64_t objid, const char *classname, Version_t version);
   virtual ~TSQLObjectInfo();

   Long64_t GetObjId() const { return fObjId; }
   const char *GetObjClassName() const { return fClassName.Data(); }
   Version_t GetObjVersion() const { return fVersion; }

protected:
   Long64_t fObjId;
   TString fClassName;
   Version_t fVersion;

   ClassDef(TSQLObjectInfo, 1) // Info (classname, version) about object in database
};

//=======================================================================

class TSQLObjectData : public TObject {

public:
   TSQLObjectData();

   TSQLObjectData(TSQLClassInfo *sqlinfo, Long64_t objid, TSQLResult *classdata, TSQLRow *classrow,
                  TSQLResult *blobdata, TSQLStatement *blobstmt);

   virtual ~TSQLObjectData();

   Long64_t GetObjId() const { return fObjId; }
   TSQLClassInfo *GetInfo() const { return fInfo; }

   Bool_t LocateColumn(const char *colname, Bool_t isblob = kFALSE);
   Bool_t IsBlobData() const { return fCurrentBlob || (fUnpack != 0); }
   void ShiftToNextValue();

   void AddUnpack(const char *tname, const char *value);
   void AddUnpackInt(const char *tname, Int_t value);

   const char *GetValue() const { return fLocatedValue; }
   const char *GetLocatedField() const { return fLocatedField; }
   const char *GetBlobPrefixName() const { return fBlobPrefixName; }
   const char *GetBlobTypeName() const { return fBlobTypeName; }

   Bool_t VerifyDataType(const char *tname, Bool_t errormsg = kTRUE);
   Bool_t PrepareForRawData();

protected:
   Bool_t ExtractBlobValues();
   Bool_t ShiftBlobRow();

   Int_t GetNumClassFields();
   const char *GetClassFieldName(Int_t n);

   TSQLClassInfo *fInfo;        //!
   Long64_t fObjId;             //!
   Bool_t fOwner;               //!
   TSQLResult *fClassData;      //!
   TSQLResult *fBlobData;       //!
   TSQLStatement *fBlobStmt;    //!
   Int_t fLocatedColumn;        //!
   TSQLRow *fClassRow;          //!
   TSQLRow *fBlobRow;           //!
   const char *fLocatedField;   //!
   const char *fLocatedValue;   //!
   Bool_t fCurrentBlob;         //!
   const char *fBlobPrefixName; ///<! name prefix in current blob row
   const char *fBlobTypeName;   ///<! name type (without prefix) in current blob row
   TObjArray *fUnpack;          //!

   ClassDef(TSQLObjectData, 1) // Keeps the data requested from the SQL server for an object.
};

// ======================================================================
/**
\class TSQLObjectDataPool
\ingroup IO
XML object keeper class
*/

class TSQLObjectDataPool : public TObject {

public:
   TSQLObjectDataPool();
   TSQLObjectDataPool(TSQLClassInfo *info, TSQLResult *data);
   virtual ~TSQLObjectDataPool();

   TSQLClassInfo *GetSqlInfo() const { return fInfo; }
   TSQLResult *GetClassData() const { return fClassData; }
   TSQLRow *GetObjectRow(Long64_t objid);

protected:
   TSQLClassInfo *fInfo;   ///<!  classinfo, for which pool is created
   TSQLResult *fClassData; ///<!  results with request to selected table
   Bool_t fIsMoreRows;     ///<!  indicates if class data has not yet read rows
   TList *fRowsPool;       ///<!  pool of extracted, but didnot used rows

   ClassDef(TSQLObjectDataPool, 1) // XML object keeper class
};

#endif
