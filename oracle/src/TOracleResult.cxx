// @(#)root/oracle:$Name:  $:$Id: TOracleResult.cxx,v 1.4 2005/04/25 17:21:11 rdm Exp $
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TOracleResult.h"
#include "TOracleRow.h"

#include <Riostream.h>

using namespace std;


ClassImp(TOracleResult)

//______________________________________________________________________________
void TOracleResult::GetMetaDataInfo()
{
   // Set fFieldInfo, fFieldCount, and fRowCount.

   if (!fResult)
      return;

   try {
      fFieldInfo = new vector<MetaData>(fResult->getColumnListMetaData());
   } catch (SQLException &oraex) {
      Error("GetMetaDataInfo()", (oraex.getMessage()).c_str());
      MakeZombie();
   }

   fFieldCount = fFieldInfo->size();
   fRowCount = -1; //doesn't provide row count
}

//______________________________________________________________________________
void TOracleResult::initResultSet(Statement *stmt)
{
   // Oracle query result.

   if (!stmt) {
      Error("initResultSet()", "construction: empty statement");
      fResultType = -1;
   } else {
      try {
         fStmt = stmt;
         if (stmt->status() == Statement::RESULT_SET_AVAILABLE) {
            fResultType  = 1;
            fResult      = stmt->getResultSet();
            GetMetaDataInfo();
            fUpdateCount = 0;
            //printf("type:%d columnsize:%d \n", fResultType, fFieldCount);
         } else if (stmt->status() == Statement::UPDATE_COUNT_AVAILABLE) {
            fResultType  = 0;
            fResult      = 0;
            fRowCount    = 0;
            fFieldInfo   = 0;
            fFieldCount  = 0;
            fUpdateCount = stmt->getUpdateCount();
         } else {
            fResultType = -1;
         }
      } catch (SQLException &oraex) {
         Error("initResultSet()", (oraex.getMessage()).c_str());
         MakeZombie();
      }
   }
}

//______________________________________________________________________________
TOracleResult::TOracleResult(Statement *stmt)
{
   initResultSet(stmt);
}

//______________________________________________________________________________
TOracleResult::TOracleResult(Statement *stmt, int row_count)
{
   initResultSet(stmt);
   // override fRowCount set by initResultSet()
   fRowCount = (row_count==-1) ? 0 : row_count;
}

//______________________________________________________________________________
TOracleResult::TOracleResult(Connection *conn, const char *tableName)
{
   // This construction func is only used to get table metainfo.

   if (!tableName || !conn) {
      Error("TOracleResult", "construction: empty input parameter");
      fResultType = -1;
   } else {
      MetaData connMD = conn->getMetaData(tableName, MetaData::PTYPE_TABLE);
      fFieldInfo   = new vector<MetaData>(connMD.getVector(MetaData::ATTR_LIST_COLUMNS));
      fFieldCount  = fFieldInfo->size();
      fRowCount    = 0;
      fResult      = 0;
      fUpdateCount = 0;
      fResultType  = 1;
   }
}

//______________________________________________________________________________
TOracleResult::~TOracleResult()
{
   // Cleanup Oracle query result.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TOracleResult::Close(Option_t *)
{
   // Close query result.

   if (!fResult || !fStmt)
      return;
   fResultType = -1;
   fStmt->closeResultSet(fResult);
   fResult    = 0;
   if (fFieldInfo)
      delete fFieldInfo;
   fRowCount  = 0;
}

//______________________________________________________________________________
Bool_t TOracleResult::IsValid(Int_t field)
{
   // Check if result set is open and field index within range.

   if (field < 0 || field >= fFieldCount) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TOracleResult::GetFieldCount()
{
   // Get number of fields in result.

   return fFieldCount;
}

//______________________________________________________________________________
const char *TOracleResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   if (!IsValid(field))
      return 0;
   string s = (*fFieldInfo)[field].getString(MetaData::ATTR_NAME);
   return (const char *)s.c_str();
}

//______________________________________________________________________________
TSQLRow *TOracleResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user.

   if (fResultType == -1) {
      Error("Next", "result set closed");
      return 0;
   }

   if (fResultType == 0) {
      // if dml query, ...
      return new TOracleRow(fUpdateCount);
   }

   // if select query,
   try {
      if (fResult->next()) {
         return new TOracleRow(fResult, fFieldInfo);
      } else
         return 0;
   } catch (SQLException &oraex) {
      Error("Next()", (oraex.getMessage()).c_str());
      MakeZombie();
   }
   return 0;
}
