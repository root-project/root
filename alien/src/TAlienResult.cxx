// @(#)root/alien:$Name:  $:$Id: TAlienResult.cxx,v 1.2 2003/09/04 10:00:00 peters Exp $
// Author: Andreas Peters 04/09/2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienResult                                                         //
//                                                                      //
// Class defining interface to an AliEn grid result.                    //
//                                                                      //
// Related class is TAlien.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienResult.h"


ClassImp(TAlienResult)

//______________________________________________________________________________
TAlienResult::TAlienResult(Grid_ResultHandle_t result)
{
   // Create a result object and initialize it with the AliEn result struct.

   fResult  = result;

   if (!gGrid) {
      Error("TAlienResult", "no instance of gGrid");
      return;
   }

   if (fResult) {
      gGrid->ResetResult(fResult);
      while (gGrid->ReadResult(fResult))
         fResults++;
      gGrid->ResetResult(fResult);
   }
}

//______________________________________________________________________________
TAlienResult::~TAlienResult()
{
   // Clean up alien guery result.

   Close();
}

//______________________________________________________________________________
void TAlienResult::Close()
{
   // Close result object.

   if (!gGrid) {
      Error("Close", "no instance of gGrid");
      return;
   }

   if (fResult)
      gGrid->CloseResult(fResult);
   fResult = 0;

   TGridResult::Close();
}

//______________________________________________________________________________
Grid_Result_t *TAlienResult::Next()
{
   // Returns next result. Returns 0 when end of result set is reached.

   if (!gGrid) {
      Error("Next", "no instance of gGrid");
      return 0;
   }

   if (!fResult)
      return 0;

   fCurrent++;
   return gGrid->ReadResult(fResult);
}

//______________________________________________________________________________
void TAlienResult::Reset()
{
   // Reset result iterator, i.e. Next() returns first result.

   if (!gGrid) {
      Error("Reset", "no instance of gGrid");
      return;
   }

   if (fResult)
      gGrid->ResetResult(fResult);
   fCurrent = 0;
}

//______________________________________________________________________________
void TAlienResult::List(Int_t indentation)
{
   // List contents of result.

   Reset();
   int cnt = 0;
   Grid_Result_t *result;
   while ((result = (Grid_Result_t *) Next())) {
      cnt++;
      for (int i = 0; i < indentation; i++)
         printf("   ");
      if (indentation)
         printf("     - %-32s %-32s\n", result->name.c_str(),
                result->name2.c_str());
      else
         printf(" [%2d] %-32s %-32s\n", cnt, result->name.c_str(),
                result->name2.c_str());
      if (result->data) {
         TAlienResult subresult(result->data);
         subresult.List(indentation + 1);
      }
   }
}
