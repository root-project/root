// @(#)root/net:$Name:  $:$Id: TGridResult.h,v 1.1.1.1 2004/09/28 14:24:59 apeters Exp $
// Author: Fons Rademakers   3/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridResult
#define ROOT_TGridResult

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridResult                                                          //
//                                                                      //
// Abstract base class defining interface to a GRID result.             //
// Objects of this class are created by TGrid methods.                  //
//                                                                      //
// Related classes are TGrid.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TList
#include "TList.h"
#endif


class TGridResult : public TList {

public:
   TGridResult() : TList() { }
   virtual ~TGridResult() { }

   ClassDef(TGridResult,1)  // ABC defining interface to GRID result set
};

#endif
