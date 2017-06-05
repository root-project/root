// @(#)root/pgsql:$Id$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPgSQLRow.h"


ClassImp(TPgSQLRow);

////////////////////////////////////////////////////////////////////////////////
/// Single row of query result.

TPgSQLRow::TPgSQLRow(void *res, ULong_t rowHandle)
{
   fResult = (PGresult *) res;
   fRowNum = (ULong_t) rowHandle;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy row object.

TPgSQLRow::~TPgSQLRow()
{
   if (fRowNum)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close row.

void TPgSQLRow::Close(Option_t *)
{
   if (!fRowNum)
      return;

   fResult = 0;
   fRowNum = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if row is open and field index within range.

Bool_t TPgSQLRow::IsValid(Int_t field)
{
   if (field < 0 || field >= (Int_t)PQnfields(fResult)) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get length in bytes of specified field.

ULong_t TPgSQLRow::GetFieldLength(Int_t field)
{
   if (!IsValid(field))
      return 0;

   ULong_t fieldLength = (ULong_t) PQfsize(fResult, field);

   if (!fieldLength) {
      Error("GetFieldLength", "cannot get field length");
      return 0;
   }

   return fieldLength;
}

////////////////////////////////////////////////////////////////////////////////
/// Get specified field from row (0 <= field < GetFieldCount()).

const char *TPgSQLRow::GetField(Int_t field)
{
   if (!IsValid(field))
      return 0;

   return PQgetvalue(fResult, fRowNum, field);
}
