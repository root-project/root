// @(#)root/sqlite:$Id$
// Author: o.freyermuth <o.f@cern.ch>, 01/06/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSQLiteResult.h"
#include "TSQLiteRow.h"

ClassImp(TSQLiteResult)

//______________________________________________________________________________
TSQLiteResult::TSQLiteResult(void *result)
{
   // SQLite query result.

   fResult     = (sqlite3_stmt *) result;

   // RowCount is -1, as sqlite cannot determine RowCount beforehand:
   fRowCount = -1;
}

//______________________________________________________________________________
TSQLiteResult::~TSQLiteResult()
{
   // Cleanup SQLite query result.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TSQLiteResult::Close(Option_t *)
{
   // Close query result.

   if (!fResult)
      return;

   sqlite3_finalize(fResult);
   fResult     = 0;
}

//______________________________________________________________________________
Bool_t TSQLiteResult::IsValid(Int_t field)
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
Int_t TSQLiteResult::GetFieldCount()
{
   // Get number of fields in result.

   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return sqlite3_column_count(fResult);
}

//______________________________________________________________________________
const char *TSQLiteResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   if (!fResult) {
      Error("GetFieldName", "result set closed");
      return 0;
   }
   return sqlite3_column_name(fResult, field);
}

//______________________________________________________________________________
TSQLRow *TSQLiteResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user.

   if (!fResult) {
      Error("Next", "result set closed");
      return 0;
   }

   int ret = sqlite3_step(fResult);
   if ((ret != SQLITE_DONE) && (ret != SQLITE_ROW)) {
      Error("Statement", "SQL Error: %d %s", ret, sqlite3_errmsg(sqlite3_db_handle(fResult)));
      return NULL;
   }
   if (ret == SQLITE_DONE) {
      // Finished executing, no other row!
      return NULL;
   }
   return new TSQLiteRow((void *) fResult, -1);
}

