// @(#)root/alien:$Name:  $:$Id: TAlienResult.h,v 1.2 2003/11/13 15:15:11 rdm Exp $
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

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif

#ifndef ROOT_TGridResult
#include "TGridResult.h"
#endif

class TAlienResult : public TGridResult {

private:
   Grid_ResultHandle_t fResult;    // AliEn result handle

public:
   TAlienResult(Grid_ResultHandle_t result);
   virtual ~TAlienResult();

   void           Close();
   const char    *GetValue() { return 0; }
   Grid_Result_t *Next();
   void           Print(Option_t *option = "") const;
   void           Reset();

   ClassDef(TAlienResult,0)    // AliEn Grid_Result result
};

#endif
