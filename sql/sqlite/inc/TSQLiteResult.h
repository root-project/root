// @(#)root/sqlite:$Id$
// Author: o.freyermuth <o.f@cern.ch>, 01/06/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLiteResult
#define ROOT_TSQLiteResult

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

#if !defined(__CINT__)
#include <sqlite3.h>
#else
struct sqlite3_stmt;
#endif


class TSQLiteResult : public TSQLResult {

private:
   sqlite3_stmt   *fResult;  // query result (rows)

   Bool_t  IsValid(Int_t field);

public:
   TSQLiteResult(void *result);
   ~TSQLiteResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TSQLiteResult, 0)  // SQLite query result
};

#endif
