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

#include "TVirtualTableInterface.h"

#include "TString.h"

class TGSimpleTableInterface : public TVirtualTableInterface {

private:
   Double_t **fData; // Pointer to 2 dimensional array of Double_t
   UInt_t fNRows;
   UInt_t fNColumns;
   TString fBuffer;

protected:

public:
   TGSimpleTableInterface(Double_t **data, UInt_t nrows = 2,
                          UInt_t ncolumns = 2);
   virtual ~TGSimpleTableInterface();

   Double_t    GetValue(UInt_t row, UInt_t column) override;
   const char *GetValueAsString(UInt_t row, UInt_t column) override;
   const char *GetRowHeader(UInt_t row) override;
   const char *GetColumnHeader(UInt_t column) override;
   UInt_t      GetNRows() override { return fNRows; }
   UInt_t      GetNColumns() override { return fNColumns; }

   ClassDefOverride(TGSimpleTableInterface, 0) // Interface to data in a 2D array of Double_t
};

#endif
