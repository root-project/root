// @(#)root/alien:$Name:$:$Id:$
// Author: Fons Rademakers   8/1/2002

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
// Class defining interface to an AliEn query result.                   //
//                                                                      //
// Related class is TAlien.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TAlienResult.h"


ClassImp(TAlienResult)

//______________________________________________________________________________
TAlienResult::TAlienResult(AlienResult_t *result)
{
   // Create result object and initialize it with the alien result struct.

   fResult  = result;
   fCurrent = 0;
   fResults = 0;

   if (fResult) {
      while (AlienFetchResult(fResult))
         fResults++;
      AlienResetResult(fResult);
   }
}

//______________________________________________________________________________
TAlienResult::~TAlienResult()
{
   // Clean up alien guery result.

   Close();
}

//______________________________________________________________________________
void TAlienResult::Close(Option_t *option)
{
   // Close result object.

   if (fResult)
      AlienFreeResult(fResult);
   fResult  = 0;
   fResults = 0;
   fCurrent = 0;
}

//______________________________________________________________________________
const char *TAlienResult::Next()
{
   // Returns next result. Returns 0 when end of result set is reached.

   if (!fResult)
      return 0;

   fCurrent++;
   return AlienFetchResult(fResult);
}

//______________________________________________________________________________
void TAlienResult::Reset()
{
   // Reset result iterator, i.e. Next() returns first result.

   if (fResult)
      AlienResetResult(fResult);
   fCurrent = 0;
}
