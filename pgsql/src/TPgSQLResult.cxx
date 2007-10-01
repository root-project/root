// @(#)root/pgsql:$Id$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPgSQLResult.h"
#include "TPgSQLRow.h"


ClassImp(TPgSQLResult)

//______________________________________________________________________________
TPgSQLResult::TPgSQLResult(void *result)
{
   // PgSQL query result.

   fResult     = (PGresult *) result;
   fRowCount   = fResult ? PQntuples(fResult) : 0;
   fCurrentRow = 0;
}

//______________________________________________________________________________
TPgSQLResult::~TPgSQLResult()
{
   // Cleanup PgSQL query result.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TPgSQLResult::Close(Option_t *)
{
   // Close query result.

   if (!fResult)
      return;

   PQclear(fResult);
   fResult     = 0;
   fRowCount   = 0;
   fCurrentRow = 0;
}

//______________________________________________________________________________
Bool_t TPgSQLResult::IsValid(Int_t field)
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
Int_t TPgSQLResult::GetFieldCount()
{
   // Get number of fields in result.

   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return PQnfields(fResult);
}

//______________________________________________________________________________
const char *TPgSQLResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   if (!fResult) {
      Error("GetFieldName", "result set closed");
      return 0;
   }
   return PQfname(fResult, field);
}

//______________________________________________________________________________
TSQLRow *TPgSQLResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user.

   Int_t row;

   if (!fResult) {
      Error("Next", "result set closed");
      return 0;
   }
   row = fCurrentRow++;
   if (row >= fRowCount)
      return 0;
   else
      return new TPgSQLRow((void *) fResult, (ULong_t) row);
}
