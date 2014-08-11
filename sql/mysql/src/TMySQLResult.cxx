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


ClassImp(TMySQLResult)

//______________________________________________________________________________
TMySQLResult::TMySQLResult(void *result)
{
   // MySQL query result.

   fResult    = (MYSQL_RES *) result;
   fRowCount  = fResult ? mysql_num_rows(fResult) : 0;
   fFieldInfo = 0;
}

//______________________________________________________________________________
TMySQLResult::~TMySQLResult()
{
   // Cleanup MySQL query result.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TMySQLResult::Close(Option_t *)
{
   // Close query result.

   if (!fResult)
      return;

   mysql_free_result(fResult);
   fResult    = 0;
   fFieldInfo = 0;
   fRowCount  = 0;
}

//______________________________________________________________________________
Bool_t TMySQLResult::IsValid(Int_t field)
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
Int_t TMySQLResult::GetFieldCount()
{
   // Get number of fields in result.

   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return mysql_num_fields(fResult);
}

//______________________________________________________________________________
const char *TMySQLResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   if (!IsValid(field))
      return 0;

   if (!fFieldInfo)
      fFieldInfo = mysql_fetch_fields(fResult);

   if (!fFieldInfo) {
      Error("GetFieldName", "cannot get field info");
      return 0;
   }

   return fFieldInfo[field].name;
}

//______________________________________________________________________________
TSQLRow *TMySQLResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user.

   MYSQL_ROW row;

   if (!fResult) {
      Error("Next", "result set closed");
      return 0;
   }
   row = mysql_fetch_row(fResult);
   if (!row)
      return 0;
   else
      return new TMySQLRow((void *) fResult, (ULong_t) row);
}
