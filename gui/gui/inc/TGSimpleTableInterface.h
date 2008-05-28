// Author: Roel Aaij   21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSimpleTableInterface
#define ROOT_TGSimpleTableInterface

#ifndef ROOT_TVirtualTableInterface
#include "TVirtualTableInterface.h"
#endif

class TGSimpleTableInterface : public TVirtualTableInterface {

private:
   Double_t **fData; // Pointer to 2 dimensional array of Double_t
   UInt_t fNRows;
   UInt_t fNColumns;

protected:

public:
   TGSimpleTableInterface(Double_t **data, UInt_t nrows = 2, 
                          UInt_t ncolumns = 2);
   virtual ~TGSimpleTableInterface();

   virtual Double_t    GetValue(UInt_t row, UInt_t column);
   virtual const char *GetValueAsString(UInt_t row, UInt_t column);
   virtual const char *GetRowHeader(UInt_t row);
   virtual const char *GetColumnHeader(UInt_t column); 
   virtual UInt_t      GetNRows() { return fNRows; }
   virtual UInt_t      GetNColumns() { return fNColumns; }

   ClassDef(TGSimpleTableInterface, 0) // Interface to data in a 2D array of Double_t
};

#endif
