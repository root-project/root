// @(#)root/mysql:$Id$
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMySQLRow.h"


ClassImp(TMySQLRow);

////////////////////////////////////////////////////////////////////////////////
/// Single row of query result.

TMySQLRow::TMySQLRow(void *res, ULong_t rowHandle)
{
   fResult      = (MYSQL_RES *) res;
   fFields      = (MYSQL_ROW) rowHandle;
   fFieldLength = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy row object.

TMySQLRow::~TMySQLRow()
{
   if (fFields)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close row.

void TMySQLRow::Close(Option_t *)
{
   if (!fFields)
      return;

   fFields      = nullptr;
   fResult      = nullptr;
   fFieldLength = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if row is open and field index within range.

Bool_t TMySQLRow::IsValid(Int_t field)
{
   if (!fFields) {
      Error("IsValid", "row closed");
      return kFALSE;
   }
   if (field < 0 || field >= (Int_t)mysql_num_fields(fResult)) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get length in bytes of specified field.

ULong_t TMySQLRow::GetFieldLength(Int_t field)
{
   if (!IsValid(field))
      return 0;

   if (!fFieldLength)
      fFieldLength = mysql_fetch_lengths(fResult);

   if (!fFieldLength) {
      Error("GetFieldLength", "cannot get field length");
      return 0;
   }

   return fFieldLength[field];
}

////////////////////////////////////////////////////////////////////////////////
/// Get specified field from row (0 <= field < GetFieldCount()).

const char *TMySQLRow::GetField(Int_t field)
{
   if (!IsValid(field))
      return nullptr;

   return fFields[field];
}
