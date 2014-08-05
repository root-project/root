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


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TSQLClassInfo keeps table information relevant for one class        //
//                                                                     //
/////////////////////////////////////////////////////////////////////////



#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

class TObjArray;

class TSQLClassColumnInfo : public TObject {

public:
   TSQLClassColumnInfo();
   TSQLClassColumnInfo(const char* name,
                       const char* sqlname,
                       const char* sqltype);
   virtual ~TSQLClassColumnInfo();

   virtual const char* GetName() const { return fName.Data(); }
   const char* GetSQLName() const { return fSQLName.Data(); }
   const char* GetSQLType() const { return fSQLType.Data(); }

protected:
   TString   fName;
   TString   fSQLName;
   TString   fSQLType;

   ClassDef(TSQLClassColumnInfo, 1); //  Keeps information about single column in class table
};

//_________________________________________________________________________________

class TSQLClassInfo : public TObject {
public:
   TSQLClassInfo();
   TSQLClassInfo(Long64_t classid,
                 const char* classname,
                 Int_t version);
   virtual ~TSQLClassInfo();


   Long64_t GetClassId() const { return fClassId; }

   virtual const char* GetName() const { return fClassName.Data(); }
   Int_t GetClassVersion() const { return fClassVersion; }

   void SetClassTableName(const char* name) { fClassTable = name; }
   void SetRawTableName(const char* name) { fRawTable = name; }

   const char* GetClassTableName() const { return fClassTable.Data(); }
   const char* GetRawTableName() const { return fRawTable.Data(); }

   void SetTableStatus(TObjArray* columns = 0, Bool_t israwtable = kFALSE);
   void SetColumns(TObjArray* columns);
   void SetRawExist(Bool_t on) { fRawtableExist = on; }

   Bool_t IsClassTableExist() const { return GetColumns()!=0; }
   Bool_t IsRawTableExist() const { return fRawtableExist; }

   TObjArray* GetColumns() const { return fColumns; }
   Int_t FindColumn(const char* name, Bool_t sqlname = kFALSE);

protected:

   TString    fClassName;            //! class name
   Int_t      fClassVersion;         //! class version
   Long64_t      fClassId;              //! sql class id
   TString    fClassTable;           //! name of table with class data
   TString    fRawTable;             //! name of table with raw data
   TObjArray* fColumns;              //! name and type of columns - array of TNamed
   Bool_t     fRawtableExist;        //! indicate that raw table is exist

   ClassDef(TSQLClassInfo, 1); //  Keeps the table information relevant for one class
};

#endif
