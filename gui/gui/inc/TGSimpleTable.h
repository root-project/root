// Author: Roel Aaij   21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSimpleTable
#define ROOT_TGSimpleTable

#include "TGTable.h"

class TGSimpleTable : public TGTable {

public:
   TGSimpleTable(TGWindow *p, Int_t id, Double_t **data,
                 UInt_t nrows, UInt_t ncolumns);
   virtual ~TGSimpleTable();

   ClassDefOverride(TGSimpleTable, 0) // A simple table that owns it's interface.
};

#endif
