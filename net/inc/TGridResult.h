// @(#)root/net:$Name:  $:$Id: TGridResult.h,v 1.2 2003/11/13 15:15:11 rdm Exp $
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
// Abstract base class defining interface to a GRID query result.       //
// Objects of this class are created by TGrid methods.                  //
//                                                                      //
// Related classes are TGrid.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TGrid
#include "TGrid.h"
#endif


class TGridResult : public TObject {

protected:
   Int_t fResults;      // number of result items
   Int_t fCurrent;      // current result item, used by Next()

   TGridResult() : fResults(0), fCurrent(0) { }

public:
   TGridResult(Grid_ResultHandle_t /*handle*/) { }
   virtual ~TGridResult() { }

   virtual void           Close() { fResults = 0; fCurrent = 0; }
   Int_t                  GetCurrent() const { return fCurrent; }
   Int_t                  GetResultCount() const { return fResults; }
   virtual const char    *GetValue() { return 0; }
   virtual Grid_Result_t *Next() { return 0; }
   virtual void           Print(Option_t * = "") const { }
   virtual void           Reset() { }

   ClassDef(TGridResult,0)  // ABC defining interface to GRID query result
};

#endif
