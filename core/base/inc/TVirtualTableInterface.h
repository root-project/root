// Author: Roel Aaij   21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualTableInterface
#define ROOT_TVirtualTableInterface

#include "Rtypes.h"


class TVirtualTableInterface {

public:
   TVirtualTableInterface() {;}
   virtual ~TVirtualTableInterface() {;}

   virtual Double_t    GetValue(UInt_t row, UInt_t column) = 0;
   virtual const char *GetValueAsString(UInt_t row, UInt_t column) = 0;
   virtual const char *GetRowHeader(UInt_t row) = 0;
   virtual const char *GetColumnHeader(UInt_t column) = 0;
   virtual UInt_t      GetNRows() = 0;
   virtual UInt_t      GetNColumns() = 0;

   ClassDef(TVirtualTableInterface, 0)
};

#endif
