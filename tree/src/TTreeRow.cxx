// @(#)root/tree:$Name$:$Id$
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
// TTreeRow                                                             //
//                                                                      //
// Class defining interface to a row of a TTree query result.           //
// Objects of this class are created by TTreeResult methods.            //
//                                                                      //
// Related classes are TTreeResult.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTreeRow.h"
#include "TString.h"
#include "TObjArray.h"


ClassImp(TTreeRow)

//______________________________________________________________________________
TTreeRow::TTreeRow(Int_t nfields)
{
   // Single row of a query result.

   fColumnCount = nfields;
   fFields      = new TString [nfields];
   fOriginal    = 0;
}

//______________________________________________________________________________
TTreeRow::TTreeRow(TSQLRow *original)
{
   // This is a shallow copy of a real row, i.e. it only contains
   // a pointer to the original.

   fFields      = 0;
   fOriginal    = 0;
   fColumnCount = 0;

   if (!original) {
      Error("TTreeRow", "original may not be 0");
      return;
   }
   if (original->IsA() != TTreeRow::Class()) {
      Error("TTreeRow", "original must be a TTreeRow");
      return;
   }

   fOriginal = (TTreeRow*) original;
   fColumnCount = fOriginal->fColumnCount;
}

//______________________________________________________________________________
TTreeRow::~TTreeRow()
{
   // Destroy row object.

   if (fFields)
      Close();
}

//______________________________________________________________________________
void TTreeRow::Close(Option_t *)
{
   // Close row.

   fOriginal = 0;

   if (!fFields)
      return;

   delete [] fFields;
   fColumnCount = 0;
}

//______________________________________________________________________________
Bool_t TTreeRow::IsValid(Int_t field)
{
   // Check if row is open and field index within range.

   if (!fFields && !fOriginal) {
      Error("IsValid", "row closed");
      return kFALSE;
   }
   if (field < 0 || field >= fColumnCount) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
ULong_t TTreeRow::GetFieldLength(Int_t field)
{
   // Get length in bytes of specified field.

   if (!IsValid(field))
      return 0;

   if (fOriginal)
      return fOriginal->fFields[field].Length();

   return fFields[field].Length();
}

//______________________________________________________________________________
const char *TTreeRow::GetField(Int_t field)
{
   // Get specified field from row (0 <= field < GetFieldCount()).

   if (!IsValid(field))
      return 0;

   if (fOriginal)
      return fOriginal->fFields[field].Data();

   return fFields[field].Data();
}

//______________________________________________________________________________
void TTreeRow::AddField(Int_t field, const char *fieldvalue)
{
   // Add field value to row. This is an internal method that is not
   // exported via the abstract interface and that should not be user called.

   if (!IsValid(field) || fOriginal)
      return;

   fFields[field] = fieldvalue;
}
