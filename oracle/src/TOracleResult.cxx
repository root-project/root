// @(#)root/oracle:$Name: v4-00-08 $:$Id: TOracleResult.cxx,v 1.0 2004/12/04 17:00:45 rdm Exp $
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


ClassImp(TOracleResult)

//______________________________________________________________________________
void TOracleResult::GetMetaDataInfo()
{
   // Set fFieldInfo, fFieldCount, and fRowCount?

   if (!fResult) 
      return;
   fFieldInfo = new vector<MetaData>(fResult->getColumnListMetaData());
   fFieldCount = fFieldInfo->size();
   fRowCount = -1; //doesn't provide row count
}

//______________________________________________________________________________
TOracleResult::TOracleResult(Statement *stmt)
{
   // Oracle query result.

   if (!stmt) {
      Error("TOracleResult", "construction: empty statement");
      fResultType = -1;
   } else {
      fStmt         = stmt;      
      if (stmt->status() == Statement::RESULT_SET_AVAILABLE) {
         fResultType = 1;
         fResult    = stmt->getResultSet();
         GetMetaDataInfo();
         fUpdateCount = 0;
         printf("type:%d columnsize:%d \n", fResultType, fFieldCount);
      } else if (stmt->status() == Statement::UPDATE_COUNT_AVAILABLE) {
         fResultType = 0;
         fResult    = 0;
         fRowCount  = 0;
         fFieldInfo = 0;
         fFieldCount= 0;
         fUpdateCount = stmt->getUpdateCount();
      } else {
         fResultType = -1;
      }
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
   fFieldInfo = 0;
   fRowCount  = 0;
}

//______________________________________________________________________________
Bool_t TOracleResult::IsValid(Int_t field)
{
   // Check if result set is open and field index within range.
   
   if (!fResult) {
      Error("IsValid", "result set closed");
      return kFALSE;
   }
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
   
   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
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
   fResult->next();
   return new TOracleRow(fResult, fFieldInfo);
}
