// @(#)root/net:$Name:$:$Id:$
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



class TGridResult : public TObject {

protected:
   Int_t    fResults;   // number of result items
   Int_t    fCurrent;   // current result item, used by Next()

   TGridResult() { fResults = fCurrent = 0; }

public:
   virtual ~TGridResult() { }

   virtual void        Close(Option_t *option="") = 0;
   Int_t               GetCurrent() const { return fCurrent; }
   Int_t               GetResultCount() const { return fResults; }
   virtual const char *GetValue() = 0;
   virtual const char *Next() = 0;
   virtual void        Reset() = 0;

   ClassDef(TGridResult,0)  // ABC defining interface to GRID query result
};

#endif
