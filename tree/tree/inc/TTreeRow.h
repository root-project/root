// @(#)root/tree:$Id$
// Author: Fons Rademakers   30/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeRow
#define ROOT_TTreeRow


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

#include "TSQLRow.h"

class TTreeRow : public TSQLRow {

friend class TTreeResult;
friend class TTreePlayer;

private:
   Int_t        fColumnCount;  ///< number of columns in row
   Int_t       *fFields;       ///<[fColumnCount] index in fRow of the end of each field
   char        *fRow;          ///< string with all the fColumnCount fields
   TTreeRow    *fOriginal;     ///<! pointer to original row

   TTreeRow(TSQLRow *original);
   Bool_t  IsValid(Int_t field);

   TTreeRow(const TTreeRow&);            // Not implemented.
   TTreeRow &operator=(const TTreeRow&); // Not implemented.

public:
   TTreeRow();
   TTreeRow(Int_t nfields);
   TTreeRow(Int_t nfields, const Int_t *fields, const char *row);
   virtual ~TTreeRow();

   void        Close(Option_t *option="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);
   void        SetRow(const Int_t *fields, const char *row);

   ClassDef(TTreeRow,1)  // One row of a TTree query result
};

#endif
