// @(#)root/sql:$Id$
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TSQLClassInfo
\ingroup IO

Contains information about tables specific to one class and
version. It provides names of table for that class. For each version of
class not more than two tables can exists. Normal table has typically
name like TH1_ver4 and additional table has name like TH1_raw4.
List of this objects are kept by TSQLFile class.
*/

#include "TSQLClassInfo.h"

#include "TObjArray.h"

ClassImp(TSQLClassColumnInfo);

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

TSQLClassColumnInfo::TSQLClassColumnInfo(const char *name, const char *sqlname, const char *sqltype)
   : TObject(), fName(name), fSQLName(sqlname), fSQLType(sqltype)
{
}

ClassImp(TSQLClassInfo);


////////////////////////////////////////////////////////////////////////////////
/// normal constructor of TSQLClassInfo class
/// Sets names of tables, which are used for that version of class

TSQLClassInfo::TSQLClassInfo(Long64_t classid, const char *classname, Int_t version)
   : TObject(), fClassName(classname), fClassVersion(version), fClassId(classid)
{
   fClassTable.Form("%s_ver%d", classname, version);
   fRawTable.Form("%s_raw%d", classname, version);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TSQLClassInfo::~TSQLClassInfo()
{
   SetColumns(nullptr);
}

////////////////////////////////////////////////////////////////////////////////
/// assigns new list of columns

void TSQLClassInfo::SetColumns(TObjArray *columns)
{
   if (fColumns) {
      fColumns->Delete();
      delete fColumns;
   }
   fColumns = columns;
}

////////////////////////////////////////////////////////////////////////////////
/// set current status of class tables

void TSQLClassInfo::SetTableStatus(TObjArray *columns, Bool_t israwtable)
{
   SetColumns(columns);
   fRawtableExist = israwtable;
}

////////////////////////////////////////////////////////////////////////////////
/// Search for column of that name
///
/// Can search either for full column name (sqlname = kFALSE, default)
/// or for name, used as column name (sqlname = kTRUE)
/// Return index of column in list (-1 if not found)

Int_t TSQLClassInfo::FindColumn(const char *name, Bool_t sqlname)
{
   if (!name || !fColumns)
      return -1;

   TIter next(fColumns);

   TSQLClassColumnInfo *col = nullptr;

   Int_t indx = 0;

   while ((col = (TSQLClassColumnInfo *)next()) != nullptr) {
      const char *colname = sqlname ? col->GetSQLName() : col->GetName();
      if (strcmp(colname, name) == 0)
         return indx;
      indx++;
   }

   return -1;
}
