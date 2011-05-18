// @(#)root/net:$Id$
// Author: Fons Rademakers   25/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLRow
#define ROOT_TSQLRow


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLRow                                                              //
//                                                                      //
// Abstract base class defining interface to a row of a SQL query       //
// result. Objects of this class are created by TSQLResult methods.     //
//                                                                      //
// Related classes are TSQLServer and TSQLResult.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TSQLRow : public TObject {

protected:
   TSQLRow() { }

public:
   virtual ~TSQLRow() { }

   virtual void        Close(Option_t *option="") = 0;
   virtual ULong_t     GetFieldLength(Int_t field) = 0;
   virtual const char *GetField(Int_t field) = 0;
   const char         *operator[](Int_t field) { return GetField(field); }

   ClassDef(TSQLRow,0)  // One row of an SQL query result
};

#endif
