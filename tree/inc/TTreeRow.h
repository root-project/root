// @(#)root/tree:$Name$:$Id$
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

#ifndef ROOT_TSQLRow
#include "TSQLRow.h"
#endif

class TString;


class TTreeRow : public TSQLRow {

friend class TTreeResult;

private:
   Int_t        fColumnCount;  // number of columns in row
   TString     *fFields;       // array containing result strings
   TTreeRow    *fOriginal;     // pointer to original row

   Bool_t  IsValid(Int_t field);
   TTreeRow(TSQLRow *original);

public:
   TTreeRow(Int_t nfields);
   virtual ~TTreeRow();

   void        AddField(Int_t field, const char *fieldvalue);

   void        Close(Option_t *option="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);

   ClassDef(TTreeRow,0)  // One row of an TTree query result
};

#endif
