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

#include <sqlite3.h>


ClassImp(TSQLiteRow);

////////////////////////////////////////////////////////////////////////////////
/// Single row of query result.

TSQLiteRow::TSQLiteRow(void *res, ULong_t /*rowHandle*/)
{
   fResult = (sqlite3_stmt *) res;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy row object.

TSQLiteRow::~TSQLiteRow()
{
   if (fResult)
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close row.

void TSQLiteRow::Close(Option_t *)
{
   fResult = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if row is open and field index within range.

Bool_t TSQLiteRow::IsValid(Int_t field)
{
   if (field < 0 || field >= (Int_t)sqlite3_column_count(fResult)) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get length in bytes of specified field.

ULong_t TSQLiteRow::GetFieldLength(Int_t field)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Get specified field from row (0 <= field < GetFieldCount()).

const char *TSQLiteRow::GetField(Int_t field)
{
   if (!IsValid(field))
      return nullptr;

   return reinterpret_cast<const char*>(sqlite3_column_text(fResult, field));
}

