// @(#)root/net:$Id$
// Author: Fons Rademakers   25/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLResult
#define ROOT_TSQLResult


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLResult                                                           //
//                                                                      //
// Abstract base class defining interface to a SQL query result.        //
// Objects of this class are created by TSQLServer methods.             //
//                                                                      //
// Related classes are TSQLServer and TSQLRow.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class  TSQLRow;


class TSQLResult : public TObject {

protected:
   Int_t    fRowCount;   // number of rows in result

   TSQLResult() : fRowCount(0) { }

public:
   virtual ~TSQLResult() { }

   virtual void        Close(Option_t *option="") = 0;
   virtual Int_t       GetFieldCount() = 0;
   virtual const char *GetFieldName(Int_t field) = 0;
   virtual Int_t       GetRowCount() const { return fRowCount; }
   virtual TSQLRow    *Next() = 0;

   ClassDef(TSQLResult,0)  // SQL query result
};

#endif
