// @(#)root/sqlite:$Id$
// Author: o.freyermuth <o.f@cern.ch>, 01/06/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSQLiteRow.h"


ClassImp(TSQLiteRow)

//______________________________________________________________________________
TSQLiteRow::TSQLiteRow(void *res, ULong_t /*rowHandle*/)
{
   // Single row of query result.

   fResult = (sqlite3_stmt *) res;
}

//______________________________________________________________________________
TSQLiteRow::~TSQLiteRow()
{
   // Destroy row object.

   if (fResult)
      Close();
}

//______________________________________________________________________________
void TSQLiteRow::Close(Option_t *)
{
   // Close row.

   fResult = 0;
}

//______________________________________________________________________________
Bool_t TSQLiteRow::IsValid(Int_t field)
{
   // Check if row is open and field index within range.

   if (field < 0 || field >= (Int_t)sqlite3_column_count(fResult)) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
ULong_t TSQLiteRow::GetFieldLength(Int_t field)
{
   // Get length in bytes of specified field.

   if (!IsValid(field))
      return 0;

   // Should call the access-method first, so sqlite3 can check whether a NULL-terminator
   // needs to be added to the byte-count, e.g. for BLOB!
   sqlite3_column_text(fResult, field);

   ULong_t fieldLength = (ULong_t) sqlite3_column_bytes(fResult, field);

   if (!fieldLength) {
      Error("GetFieldLength", "cannot get field length");
      return 0;
   }

   return fieldLength;
}

//______________________________________________________________________________
const char *TSQLiteRow::GetField(Int_t field)
{
   // Get specified field from row (0 <= field < GetFieldCount()).

   if (!IsValid(field))
      return 0;

   return reinterpret_cast<const char*>(sqlite3_column_text(fResult, field));
}

