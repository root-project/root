// @(#)root/alien:$Name:$:$Id:$
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienResult
#define ROOT_TAlienResult


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienResult                                                         //
//                                                                      //
// Class defining interface to an AliEn query result.                   //
//                                                                      //
// Related class is TAlien.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridResult
#include "TGridResult.h"
#endif

#if !defined(__CINT__)
#include <AliEn.h>
#else
struct AlienResult_t;
#endif


class TAlienResult : public TGridResult {

private:
   AlienResult_t  *fResult;    // AliEn result object

public:
   TAlienResult(AlienResult_t *result);
   ~TAlienResult();

   void        Close(Option_t *option="");
   const char *GetValue() { return 0; }
   const char *Next();
   void        Reset();

   ClassDef(TAlienResult,0)  // AliEn query result
};

#endif
