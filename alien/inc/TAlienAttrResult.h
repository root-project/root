// @(#)root/alien:$Name:$:$Id:$
// Author: Fons Rademakers   13/5/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienAttrResult
#define ROOT_TAlienAttrResult


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienAttrResult                                                     //
//                                                                      //
// Class defining interface to an AliEn attribute query result.         //
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
struct AlienAttr_t;
#endif


class TAlienAttrResult : public TGridResult {

private:
   AlienAttr_t  *fResult;    // AliEn attribute result object

public:
   TAlienAttrResult(AlienAttr_t *result);
   ~TAlienAttrResult();

   void        Close(Option_t *option="");
   const char *GetValue();
   const char *Next();
   void        Reset();

   ClassDef(TAlienAttrResult,0)  // AliEn attribute query result
};

#endif
