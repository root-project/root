// @(#)root/tree:$Id$
// Author: Fons Rademakers   30/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TTreeResult
\ingroup tree

Class defining interface to a TTree query result with the same
interface as for SQL databases. A TTreeResult is returned by
TTree::Query() (actually TTreePlayer::Query()).

Related classes are TTreeRow.
*/

#include "TTreeResult.h"
#include "TTreeRow.h"
#include "TString.h"
#include "TObjArray.h"

ClassImp(TTreeResult);

////////////////////////////////////////////////////////////////////////////////
/// Create a query result object.

TTreeResult::TTreeResult()
{
   fColumnCount = 0;
   fRowCount    = 0;
   fFields      = 0;
   fResult      = 0;
   fNextRow     = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a query result object.

TTreeResult::TTreeResult(Int_t nfields)
{
   fColumnCount = nfields;
   fRowCount    = 0;
   fFields      = new TString [nfields];
   fResult      = new TObjArray;
   fNextRow     = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup result object.

TTreeResult::~TTreeResult()
{
   if (fResult)
      Close();

   delete [] fFields;
}

////////////////////////////////////////////////////////////////////////////////
/// Close query result.

void TTreeResult::Close(Option_t *)
{
   if (!fResult)
      return;

   fResult->Delete();
   delete fResult;
   fResult   = 0;
   fRowCount = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if result set is open and field index within range.

Bool_t TTreeResult::IsValid(Int_t field)
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

Int_t TTreeResult::GetFieldCount()
{
   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return fColumnCount;
}

////////////////////////////////////////////////////////////////////////////////
/// Get name of specified field.

const char *TTreeResult::GetFieldName(Int_t field)
{
   if (!IsValid(field))
      return 0;

   return fFields[field].Data();
}

////////////////////////////////////////////////////////////////////////////////
/// Get next query result row. The returned object must be
/// deleted by the user and becomes invalid when the result set is
/// closed or deleted.

TSQLRow *TTreeResult::Next()
{
   if (!fResult) {
      Error("Next", "result set closed");
      return 0;
   }

   if (fNextRow >= fRowCount)
      return 0;
   else {
      TTreeRow *row = new TTreeRow((TTreeRow*)fResult->At(fNextRow));
      fNextRow++;
      return row;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add field name to result set. This is an internal method that is not
/// exported via the abstract interface and that should not be user called.

void TTreeResult::AddField(Int_t field, const char *fieldname)
{
   if (!IsValid(field))
      return;

   fFields[field] = fieldname;
}

////////////////////////////////////////////////////////////////////////////////
/// Adopt a row to result set. This is an internal method that is not
/// exported via the abstract interface and that should not be user called.

void TTreeResult::AddRow(TSQLRow *row)
{
   if (!fResult) {
      Error("AddRow", "result set closed");
      return;
   }

   fResult->Add(row);
   fRowCount++;
}
