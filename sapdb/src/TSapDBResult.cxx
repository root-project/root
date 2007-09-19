// @(#)root/sapdb:$Id$
// Author: Mark Hemberger & Fons Rademakers   03/08/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TString.h"
#include "TSapDBResult.h"
#include "TSapDBRow.h"


ClassImp(TSapDBResult)

//______________________________________________________________________________
TSapDBResult::TSapDBResult(SQLHSTMT result, SDWORD rowCount)
{
   // SapDB query result.

   fResult     = result;
   fFieldNames = 0;
   fFieldCount = -1;
   fRowCount   = 0;

   if (fResult) {
      SQLLEN rowcount = 0;
      if (SQLRowCount(fResult, &rowcount) != SQL_SUCCESS) {
         Error("TSapDBResult", "no rows counted");
      }
      // -1 means: result has been found but the number of columns is
      // undetermined (only for SYSTEM tables, a valid number is available)
      fRowCount = rowcount < 0 ? rowCount : rowcount;
   }
}

//______________________________________________________________________________
TSapDBResult::~TSapDBResult()
{
   // Cleanup SapDB query result.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TSapDBResult::Close(Option_t *)
{
   // Close query result.

   if (!fResult)
      return;

   delete [] fFieldNames;
   fFieldNames = 0;
   fFieldCount = 0;
   fResult     = 0;
   fRowCount   = 0;
}

//______________________________________________________________________________
Bool_t TSapDBResult::IsValid(Int_t field)
{
   // Check if result set is open and field index within range.

   if (!fResult) {
      Error("IsValid", "result set closed");
      return kFALSE;
   }
   if (field < 0 || field >= GetFieldCount()) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t TSapDBResult::GetFieldCount()
{
   // Get number of fields in result.

   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }

   if (fFieldCount >= 0)
      return fFieldCount;

   SQLSMALLINT columnCount;
   if (SQLNumResultCols(fResult, &columnCount) == SQL_SUCCESS)
      fFieldCount = columnCount;
   else
      fFieldCount = 0;

   return fFieldCount;
}

//______________________________________________________________________________
const char *TSapDBResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   if (!IsValid(field))
      return 0;

   if (!fFieldNames)
      fFieldNames = new TString[GetFieldCount()];

   if (!fFieldNames[field].IsNull())
      return fFieldNames[field];

   // Get name of specified field.
   SQLUSMALLINT columnNumber;
   SQLCHAR      columnName[256];
   SQLSMALLINT  bufferLength = 256;
   SQLSMALLINT  nameLength;
   SQLSMALLINT  dataType;
   SQLULEN      columnSize;
   SQLSMALLINT  decimalDigits;
   SQLSMALLINT  nullable;

   columnNumber = field + 1;
   if (SQLDescribeCol(fResult, columnNumber, columnName, bufferLength,
                      &nameLength, &dataType, &columnSize, &decimalDigits,
                      &nullable) == SQL_SUCCESS) {
      //printf ("ColumnNumber: %d\n", columnNumber);
      //printf ("ColumnName: %s\n", columnName);
      //printf ("DataType: %d\n", dataType);
      //printf ("ColumnSize: %ld\n", columnSize);
      fFieldNames[field] = (const char *)columnName;
   } else {
      Error("GetFieldName", "cannot get field info");
      return 0;
   }

   return fFieldNames[field];
}

//______________________________________________________________________________
TSQLRow *TSapDBResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user.

   if (!fResult) {
      Error("Next", "result set closed");
      return 0;
   }

   RETCODE rc = SQLFetchScroll(fResult, SQL_FETCH_NEXT, 0);
   if (rc == SQL_SUCCESS)
      return new TSapDBRow(fResult, GetFieldCount());
   else if (rc == SQL_NO_DATA)
      return 0;
   else {
      Error("Next", "error during fetchscroll");
      return 0;
   }

   return 0;
}
