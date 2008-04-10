// @(#)root/pgsql:$Id$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPgSQLResult
#define ROOT_TPgSQLResult

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

#if !defined(__CINT__)
#include <libpq-fe.h>
#else
struct PGresult;
#endif


class TPgSQLResult : public TSQLResult {

private:
   PGresult   *fResult;      // query result (rows)
   ULong_t     fCurrentRow;  // info to result row

   Bool_t  IsValid(Int_t field);

public:
   TPgSQLResult(void *result);
   ~TPgSQLResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TPgSQLResult, 0)  // PgSQL query result
};

#endif
