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


ClassImp(TMySQLRow)

//______________________________________________________________________________
TMySQLRow::TMySQLRow(void *res, ULong_t rowHandle)
{
   // Single row of query result.

   fResult      = (MYSQL_RES *) res;
   fFields      = (MYSQL_ROW) rowHandle;
   fFieldLength = 0;
}

//______________________________________________________________________________
TMySQLRow::~TMySQLRow()
{
   // Destroy row object.

   if (fFields)
      Close();
}

//______________________________________________________________________________
void TMySQLRow::Close(Option_t *)
{
   // Close row.
   
   if (!fFields)
      return;

   fFields      = 0;
   fResult      = 0;
   fFieldLength = 0;
}

//______________________________________________________________________________
Bool_t TMySQLRow::IsValid(Int_t field)
{
   // Check if row is open and field index within range.
   
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

//______________________________________________________________________________
ULong_t TMySQLRow::GetFieldLength(Int_t field)
{
   // Get length in bytes of specified field.
   
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

//______________________________________________________________________________
const char *TMySQLRow::GetField(Int_t field)
{
   // Get specified field from row (0 <= field < GetFieldCount()).

   if (!IsValid(field))
      return 0;

   return fFields[field];
}
