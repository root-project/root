// @(#)root/net:$Id$
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

class TEntryList;


class TGridResult : public TList {

public:
   TGridResult() : TList() { SetOwner(kTRUE); }
   virtual ~TGridResult() { }

   virtual const char *GetFileName(UInt_t) const
      { MayNotUse("GetFileName"); return 0; }
   virtual const char *GetFileNamePath(UInt_t) const
      { MayNotUse("GetFileNamePath"); return 0; }
   virtual const char *GetPath(UInt_t) const
      { MayNotUse("GetPath"); return 0; }
   virtual const TEntryList *GetEntryList(UInt_t) const
      { MayNotUse("GetEntryList"); return 0; }
   virtual const char *GetKey(UInt_t, const char*) const
      { MayNotUse("GetKey"); return 0; }
   virtual Bool_t      SetKey(UInt_t, const char*, const char*)
      { MayNotUse("SetKey"); return 0; }
   virtual TList      *GetFileInfoList() const
      { MayNotUse("GetFileInfoList"); return 0; }

   ClassDef(TGridResult,1)  // ABC defining interface to GRID result set
};

#endif
