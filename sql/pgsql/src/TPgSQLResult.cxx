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

////////////////////////////////////////////////////////////////////////////////
/// PgSQL query result.

TPgSQLResult::TPgSQLResult(void *result)
{
   fResult     = (PGresult *) result;
   fRowCount   = fResult ? PQntuples(fResult) : 0;
   fCurrentRow = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup PgSQL query result.

TPgSQLResult::~TPgSQLResult()
{
   if (fResult)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close query result.

void TPgSQLResult::Close(Option_t *)
{
   if (!fResult)
      return;

   PQclear(fResult);
   fResult     = 0;
   fRowCount   = 0;
   fCurrentRow = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if result set is open and field index within range.

Bool_t TPgSQLResult::IsValid(Int_t field)
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

Int_t TPgSQLResult::GetFieldCount()
{
   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return PQnfields(fResult);
}

////////////////////////////////////////////////////////////////////////////////
/// Get name of specified field.

const char *TPgSQLResult::GetFieldName(Int_t field)
{
   if (!fResult) {
      Error("GetFieldName", "result set closed");
      return 0;
   }
   return PQfname(fResult, field);
}

////////////////////////////////////////////////////////////////////////////////
/// Get next query result row. The returned object must be
/// deleted by the user.

TSQLRow *TPgSQLResult::Next()
{
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
