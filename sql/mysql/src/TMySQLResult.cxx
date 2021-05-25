// @(#)root/mysql:$Id$
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMySQLResult.h"
#include "TMySQLRow.h"


ClassImp(TMySQLResult);

////////////////////////////////////////////////////////////////////////////////
/// MySQL query result.

TMySQLResult::TMySQLResult(void *result)
{
   fResult    = (MYSQL_RES *) result;
   fRowCount  = fResult ? mysql_num_rows(fResult) : 0;
   fFieldInfo = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup MySQL query result.

TMySQLResult::~TMySQLResult()
{
   if (fResult)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close query result.

void TMySQLResult::Close(Option_t *)
{
   if (!fResult)
      return;

   mysql_free_result(fResult);
   fResult    = nullptr;
   fFieldInfo = nullptr;
   fRowCount  = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if result set is open and field index within range.

Bool_t TMySQLResult::IsValid(Int_t field)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get number of fields in result.

Int_t TMySQLResult::GetFieldCount()
{
   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return mysql_num_fields(fResult);
}

////////////////////////////////////////////////////////////////////////////////
/// Get name of specified field.

const char *TMySQLResult::GetFieldName(Int_t field)
{
   if (!IsValid(field))
      return nullptr;

   if (!fFieldInfo)
      fFieldInfo = mysql_fetch_fields(fResult);

   if (!fFieldInfo) {
      Error("GetFieldName", "cannot get field info");
      return nullptr;
   }

   return fFieldInfo[field].name;
}

////////////////////////////////////////////////////////////////////////////////
/// Get next query result row. The returned object must be
/// deleted by the user.

TSQLRow *TMySQLResult::Next()
{
   if (!fResult) {
      Error("Next", "result set closed");
      return nullptr;
   }
   MYSQL_ROW row = mysql_fetch_row(fResult);
   if (!row)
      return nullptr;

   return new TMySQLRow((void *) fResult, (ULong_t) row);
}
