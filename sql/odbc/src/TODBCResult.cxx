// @(#)root/odbc:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TODBCResult.h"
#include "TODBCRow.h"


ClassImp(TODBCResult)

//______________________________________________________________________________
TODBCResult::TODBCResult(SQLHSTMT stmt)
{
   // Constructor

   fHstmt = stmt;
   fFieldCount = 0;

   SQLSMALLINT   columnCount;

   SQLRETURN retcode = SQLNumResultCols(fHstmt, &columnCount);

   if (retcode == SQL_SUCCESS || retcode == SQL_SUCCESS_WITH_INFO)
      fFieldCount = columnCount;
}

//______________________________________________________________________________
TODBCResult::~TODBCResult()
{
   // Cleanup ODBC query result.

   Close();
}

//______________________________________________________________________________
void TODBCResult::Close(Option_t *)
{
   // Close (cleanup) ODBC result object. Deletes statement

   SQLFreeHandle(SQL_HANDLE_STMT, fHstmt);
   fHstmt = 0;
}

//______________________________________________________________________________
const char *TODBCResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   SQLCHAR columnName[1024];

   SQLSMALLINT nameLength;
   SQLSMALLINT dataType;
   SQLULEN     columnSize;
   SQLSMALLINT decimalDigits;
   SQLSMALLINT nullable;

   SQLRETURN retcode =
      SQLDescribeCol(fHstmt, field+1, columnName, 1024,
                     &nameLength, &dataType,
                     &columnSize, &decimalDigits, &nullable);

   if (retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO) return 0;

   fNameBuffer = (const char*) columnName;

   return fNameBuffer;
}

//______________________________________________________________________________
TSQLRow *TODBCResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user.

   if (fHstmt==0) return 0;

   SQLRETURN retcode = SQLFetch(fHstmt);

   if (retcode == SQL_SUCCESS || retcode == SQL_SUCCESS_WITH_INFO)
       return new TODBCRow(fHstmt, fFieldCount);

   return 0;
}
