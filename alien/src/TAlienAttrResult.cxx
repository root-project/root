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
// TAlienAttrResult                                                     //
//                                                                      //
// Class defining interface to an AliEn attribute query result.         //
//                                                                      //
// Related class is TAlien.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TAlienAttrResult.h"


ClassImp(TAlienAttrResult)

//______________________________________________________________________________
TAlienAttrResult::TAlienAttrResult(AlienAttr_t *result)
{
   // Create attribute query result object and initialize it with the
   // alien attribute struct.

   fResult  = result;
   fCurrent = 0;
   fResults = 0;

   if (fResult) {
      while (AlienFetchAttribute(fResult))
         fResults++;
      AlienResetAttribute(fResult);
   }
}

//______________________________________________________________________________
TAlienAttrResult::~TAlienAttrResult()
{
   // Clean up alien attribute result.

   Close();
}

//______________________________________________________________________________
void TAlienAttrResult::Close(Option_t *option)
{
   // Close result object.

   if (fResult)
      AlienFreeAttribute(fResult);
   fResult  = 0;
   fResults = 0;
   fCurrent = 0;
}

//______________________________________________________________________________
const char *TAlienAttrResult::Next()
{
   // Returns next result. Returns 0 when end of result set is reached.

   if (!fResult)
      return 0;

   fCurrent++;
   return AlienFetchAttribute(fResult);
}

//______________________________________________________________________________
const char *TAlienAttrResult::GetValue()
{
   // Returns attribute value. Call after Next() which is used to obtain
   // the attribute name. Returns 0 when end of result set is reached.

   if (!fResult)
      return 0;

   return AlienFetchValue(fResult);
}

//______________________________________________________________________________
void TAlienAttrResult::Reset()
{
   // Reset result iterator, i.e. Next() returns first result.

   if (fResult)
      AlienResetAttribute(fResult);
   fCurrent = 0;
}
