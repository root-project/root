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


ClassImp(TODBCResult);

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TODBCResult::TODBCResult(SQLHSTMT stmt)
{
   fHstmt = stmt;
   fFieldCount = 0;

   SQLSMALLINT   columnCount;

   SQLRETURN retcode = SQLNumResultCols(fHstmt, &columnCount);

   if (retcode == SQL_SUCCESS || retcode == SQL_SUCCESS_WITH_INFO)
      fFieldCount = columnCount;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup ODBC query result.

TODBCResult::~TODBCResult()
{
   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close (cleanup) ODBC result object. Deletes statement

void TODBCResult::Close(Option_t *)
{
   SQLFreeHandle(SQL_HANDLE_STMT, fHstmt);
   fHstmt = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get name of specified field.

const char *TODBCResult::GetFieldName(Int_t field)
{
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

   if (retcode != SQL_SUCCESS && retcode != SQL_SUCCESS_WITH_INFO) return nullptr;

   fNameBuffer = (const char*) columnName;

   return fNameBuffer;
}

////////////////////////////////////////////////////////////////////////////////
/// Get next query result row. The returned object must be
/// deleted by the user.

TSQLRow *TODBCResult::Next()
{
   if (!fHstmt) return nullptr;

   SQLRETURN retcode = SQLFetch(fHstmt);

   if (retcode == SQL_SUCCESS || retcode == SQL_SUCCESS_WITH_INFO)
       return new TODBCRow(fHstmt, fFieldCount);

   return nullptr;
}
