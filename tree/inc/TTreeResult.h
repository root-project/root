// @(#)root/tree:$Name$:$Id$
// Author: Fons Rademakers   30/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeResult
#define ROOT_TTreeResult


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

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

class TString;
class TObjArray;


class TTreeResult : public TSQLResult {

private:
   Int_t       fColumnCount;   // number of columns in result
   TString    *fFields;        // array containing field strings
   TObjArray  *fResult;        // query result (TTreeRow objects)
   Int_t       fNextRow;       // row iterator

   Bool_t  IsValid(Int_t field);

public:
   TTreeResult(Int_t nfields);
   virtual ~TTreeResult();

   void        AddField(Int_t field, const char *fieldname);
   void        AddRow(TSQLRow *row);

   void        Close(Option_t *option="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TTreeResult,0)  // TTree query result
};

#endif
