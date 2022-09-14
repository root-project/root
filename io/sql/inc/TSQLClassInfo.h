// @(#)root/sql:$Id$
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLClassInfo
#define ROOT_TSQLClassInfo

#include "TObject.h"

#include "TString.h"

class TObjArray;

class TSQLClassColumnInfo final : public TObject {

public:
   TSQLClassColumnInfo() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see ROOT-10300
   TSQLClassColumnInfo(const char *name, const char *sqlname, const char *sqltype);

   const char *GetName() const final { return fName.Data(); }
   const char *GetSQLName() const { return fSQLName.Data(); }
   const char *GetSQLType() const { return fSQLType.Data(); }

protected:
   TString fName;
   TString fSQLName;
   TString fSQLType;

   ClassDefOverride(TSQLClassColumnInfo, 1); //  Keeps information about single column in class table
};

//_________________________________________________________________________________

class TSQLClassInfo final : public TObject {
public:
   TSQLClassInfo() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see ROOT-10300
   TSQLClassInfo(Long64_t classid, const char *classname, Int_t version);
   virtual ~TSQLClassInfo();

   Long64_t GetClassId() const { return fClassId; }

   const char *GetName() const final { return fClassName.Data(); }
   Int_t GetClassVersion() const { return fClassVersion; }

   void SetClassTableName(const char *name) { fClassTable = name; }
   void SetRawTableName(const char *name) { fRawTable = name; }

   const char *GetClassTableName() const { return fClassTable.Data(); }
   const char *GetRawTableName() const { return fRawTable.Data(); }

   void SetTableStatus(TObjArray *columns = nullptr, Bool_t israwtable = kFALSE);
   void SetColumns(TObjArray *columns);
   void SetRawExist(Bool_t on) { fRawtableExist = on; }

   Bool_t IsClassTableExist() const { return GetColumns() != nullptr; }
   Bool_t IsRawTableExist() const { return fRawtableExist; }

   TObjArray *GetColumns() const { return fColumns; }
   Int_t FindColumn(const char *name, Bool_t sqlname = kFALSE);

protected:
   TString fClassName;             ///<! class name
   Int_t fClassVersion{0};         ///<! class version
   Long64_t fClassId{0};           ///<! sql class id
   TString fClassTable;            ///<! name of table with class data
   TString fRawTable;              ///<! name of table with raw data
   TObjArray *fColumns{nullptr};   ///<! name and type of columns - array of TNamed
   Bool_t fRawtableExist{kFALSE};  ///<! indicate that raw table is exist

   ClassDefOverride(TSQLClassInfo, 1); //  Keeps the table information relevant for one class
};

#endif
