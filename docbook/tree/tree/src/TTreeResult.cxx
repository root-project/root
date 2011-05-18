// @(#)root/tree:$Id$
// Author: Fons Rademakers   30/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeResult                                                          //
//                                                                      //
// Class defining interface to a TTree query result with the same       //
// interface as for SQL databases. A TTreeResult is returned by         //
// TTree::Query() (actually TTreePlayer::Query()).                      //
//                                                                      //
// Related classes are TTreeRow.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTreeResult.h"
#include "TTreeRow.h"
#include "TString.h"
#include "TObjArray.h"


ClassImp(TTreeResult)

//______________________________________________________________________________
TTreeResult::TTreeResult()
{
   // Create a query result object.

   fColumnCount = 0;
   fRowCount    = 0;
   fFields      = 0;
   fResult      = 0;
   fNextRow     = 0;
}

//______________________________________________________________________________
TTreeResult::TTreeResult(Int_t nfields)
{
   // Create a query result object.

   fColumnCount = nfields;
   fRowCount    = 0;
   fFields      = new TString [nfields];
   fResult      = new TObjArray;
   fNextRow     = 0;
}

//______________________________________________________________________________
TTreeResult::~TTreeResult()
{
   // Cleanup result object.

   if (fResult)
      Close();

   delete [] fFields;
}

//______________________________________________________________________________
void TTreeResult::Close(Option_t *)
{
   // Close query result.

   if (!fResult)
      return;

   fResult->Delete();
   delete fResult;
   fResult   = 0;
   fRowCount = 0;
}

//______________________________________________________________________________
Bool_t TTreeResult::IsValid(Int_t field)
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
Int_t TTreeResult::GetFieldCount()
{
   // Get number of fields in result.

   if (!fResult) {
      Error("GetFieldCount", "result set closed");
      return 0;
   }
   return fColumnCount;
}

//______________________________________________________________________________
const char *TTreeResult::GetFieldName(Int_t field)
{
   // Get name of specified field.

   if (!IsValid(field))
      return 0;

   return fFields[field].Data();
}

//______________________________________________________________________________
TSQLRow *TTreeResult::Next()
{
   // Get next query result row. The returned object must be
   // deleted by the user and becomes invalid when the result set is
   // closed or deleted.

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

//______________________________________________________________________________
void TTreeResult::AddField(Int_t field, const char *fieldname)
{
   // Add field name to result set. This is an internal method that is not
   // exported via the abstract interface and that should not be user called.

   if (!IsValid(field))
      return;

   fFields[field] = fieldname;
}

//______________________________________________________________________________
void TTreeResult::AddRow(TSQLRow *row)
{
   // Adopt a row to result set. This is an internal method that is not
   // exported via the abstract interface and that should not be user called.

   if (!fResult) {
      Error("AddRow", "result set closed");
      return;
   }

   fResult->Add(row);
   fRowCount++;
}
