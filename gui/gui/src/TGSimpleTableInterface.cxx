// Author: Roel Aaij 15/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClass.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TEnv.h"
#include "TGSimpleTableInterface.h"
#include "TGResourcePool.h"
#include "TError.h"

ClassImp(TGSimpleTableInterface)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGSimpleTableInterface                                               //
//                                                                      //
// TGSimpleTableInterface is a very simple implementation of a          //
// TVirtualTableInterface. This interface provides a TGTable with data  //
// from a two dimensional array of doubles in memory. It is mostly      //
// meant as an example implementation for a TVirtualTableInterface.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGSimpleTableInterface::TGSimpleTableInterface (Double_t **data, 
                                                UInt_t nrows, UInt_t ncolumns) 
   : TVirtualTableInterface(), fData(data), fNRows(nrows), fNColumns(ncolumns)
{
   // TGSimpleTableInterfac constructor.
}

//______________________________________________________________________________
TGSimpleTableInterface::~TGSimpleTableInterface()
{ 
   // TGSimpleTableInterface destructor.
}

//______________________________________________________________________________
Double_t TGSimpleTableInterface::GetValue(UInt_t row, UInt_t column)
{
   // Return the value of the double in row,column of the data.

   if ((row > fNRows) || (column > fNColumns)) {
      Error("TGSimpleTableInterface","Non existing value requested.");
      return 0;
   } else {
      Double_t temp = fData[row][column];
      return temp;
   }
}

//______________________________________________________________________________
const char *TGSimpleTableInterface::GetValueAsString(UInt_t row, UInt_t column)
{
   // Return the value of the double in row,column of the data as a string.

   // FIXME use template string for string format instead of hardcoded format

   return StrDup(TString::Format("%5.2f", GetValue(row, column)));
}

//______________________________________________________________________________
const char *TGSimpleTableInterface::GetRowHeader(UInt_t row)
{
   // Return a name for the header at row.

   return StrDup(TString::Format("DRow %d", row));
}

//______________________________________________________________________________
const char *TGSimpleTableInterface::GetColumnHeader(UInt_t column)
{
   // Return a name for the header at column.

   return StrDup(TString::Format("DCol %d", column));
}
