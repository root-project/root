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
#include "TSapDBRow.h"


ClassImp(TSapDBRow)

//______________________________________________________________________________
TSapDBRow::TSapDBRow(SQLHSTMT result, Int_t nfields)
{
   // Single row of query result.

   fResult      = result;
   fFieldCount  = nfields;
   fFieldLength = 0;
   fFieldValue  = 0;
}

//______________________________________________________________________________
TSapDBRow::~TSapDBRow()
{
   // Destroy row object.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TSapDBRow::Close(Option_t *)
{
   // Close row.

   delete [] fFieldLength;
   delete [] fFieldValue;
   fResult      = 0;
   fFieldCount  = 0;
   fFieldLength = 0;
   fFieldValue  = 0;
}

//______________________________________________________________________________
Bool_t TSapDBRow::IsValid(Int_t field)
{
   // Check if row is open and field index within range.

   if (field < 0 || field >= fFieldCount) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
ULong_t TSapDBRow::GetFieldLength(Int_t field)
{
   // Get length in bytes of specified field.

   if (!IsValid(field))
      return 0;

   if (!fFieldLength) {
      fFieldLength = new ULong_t[fFieldCount];
      for (int i = 0; i < fFieldCount; i++)
         fFieldLength[i] = 0;
   }

   if (fFieldLength[field])
      return fFieldLength[field];

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
      fFieldLength[field] = columnSize;
      return columnSize;
   } else {
      Error("GetFieldLength", "cannot get field length");
      return 0;
   }
}

//______________________________________________________________________________
const char *TSapDBRow::GetField(Int_t field)
{
   // Get specified field from row (0 <= field < GetFieldCount()).

   if (!IsValid(field))
      return 0;

   if (!fFieldValue)
      fFieldValue = new TString[fFieldCount];

   if (!fFieldValue[field].IsNull())
      return fFieldValue[field];

   SQLUSMALLINT columnNumber;
   SQLCHAR      columnName[256];
   SQLSMALLINT  bufferLength = 256;
   SQLSMALLINT  nameLength;
   SQLSMALLINT  dataType;
   SQLULEN      columnSize;
   SQLSMALLINT  decimalDigits;
   SQLSMALLINT  nullable;

   columnNumber = field + 1;
   RETCODE rc;
   rc = SQLDescribeCol(fResult, columnNumber, columnName, bufferLength,
                       &nameLength, &dataType, &columnSize, &decimalDigits,
                       &nullable);
   if (rc != SQL_SUCCESS) {
      Error("TSapDBRow", "error in getting description");
      return 0;
   }

   if (columnSize > 4000) {
      Error("TSapDBRow", "column size too large for current implementation.");
      return 0;
   }

   SQLLEN     strLenOrIndPtr;
   SQLPOINTER targetValuePtr[4000];
   bufferLength = sizeof(targetValuePtr);

   if (SQLGetData(fResult, columnNumber, SQL_C_DEFAULT, targetValuePtr,
                  bufferLength, &strLenOrIndPtr) != SQL_SUCCESS) {
      Error("TSapDBRow", "error in getting data");
      return 0;
   }

   char fieldstr[4001];

   switch (dataType) {
      case SQL_CHAR:
      case SQL_VARCHAR:
      case SQL_LONGVARCHAR:
      // not yet supported...
      //case SQL_WCHAR:
      //case SQL_WVARCHAR:
      //case SQL_WLONGVARCHAR:
         snprintf(fieldstr,4001, "%-*.*s", (int)columnSize, (int)columnSize,
                 (char*) targetValuePtr);
         break;
      case SQL_TINYINT:
      case SQL_SMALLINT:
      case SQL_INTEGER:
      case SQL_BIGINT:
         snprintf(fieldstr,4001, "%-*ld", (int)columnSize, *(long int*)(targetValuePtr));
         break;
      case SQL_DECIMAL:
      case SQL_NUMERIC:
      case SQL_REAL:
      case SQL_FLOAT:
         snprintf(fieldstr,4001, "%-*.2f", (int)columnSize, *(float*)(targetValuePtr));
         break;
      case SQL_DOUBLE:
         snprintf(fieldstr,4001, "%-*.2f", (int)columnSize, *(double*)(targetValuePtr));
         break;
      case SQL_BIT:
      case SQL_BINARY:
      case SQL_VARBINARY:
      case SQL_LONGVARBINARY:
      case SQL_TYPE_DATE:
      case SQL_TYPE_TIME:
      case SQL_TYPE_TIMESTAMP:
      default:
         snprintf(fieldstr,4001, "%-*.*s", (int)columnSize, (int)columnSize,
                 (char*)targetValuePtr);
         break;
   }

   fFieldValue[field] = fieldstr;

   return fFieldValue[field];
}
