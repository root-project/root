// @(#)root/pgsql:$Name:$:$Id:$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPgSQLRow.h"


ClassImp(TPgSQLRow)

//______________________________________________________________________________
TPgSQLRow::TPgSQLRow(void *res, ULong_t rowHandle)
{
   // Single row of query result.

   fResult      = (PGresult *) res;
   fRowNum      = (ULong_t) rowHandle;
   fFieldLength = 0;
}

//______________________________________________________________________________
TPgSQLRow::~TPgSQLRow()
{
   // Destroy row object.

   if (fRowNum)
      Close();
}

//______________________________________________________________________________
void TPgSQLRow::Close(Option_t *)
{
   // Close row.

   if (!fRowNum)
      return;

   fResult      = 0;
   fRowNum      = 0;
   fFieldLength = 0;
}

//______________________________________________________________________________
Bool_t TPgSQLRow::IsValid(Int_t field)
{
   // Check if row is open and field index within range.

   if (field < 0 || field >= (Int_t)PQnfields(fResult)) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
ULong_t TPgSQLRow::GetFieldLength(Int_t field)
{
   // Get length in bytes of specified field.

   if (!IsValid(field))
      return 0;

   if (!fFieldLength)
      fFieldLength = (ULong_t) PQfsize(fResult, field);

   if (!fFieldLength) {
      Error("GetFieldLength", "cannot get field length");
      return 0;
   }

   return fFieldLength;
}

//______________________________________________________________________________
const char *TPgSQLRow::GetField(Int_t field)
{
   // Get specified field from row (0 <= field < GetFieldCount()).

   if (!IsValid(field))
      return 0;

   return PQgetvalue(fResult, fRowNum, field);
}
