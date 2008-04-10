// @(#)root/pgsql:$Id$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPgSQLRow
#define ROOT_TPgSQLRow

#ifndef ROOT_TSQLRow
#include "TSQLRow.h"
#endif

#if !defined(__CINT__)
#include <libpq-fe.h>
#else
struct PGresult;
typedef char **PGresAttValue;
#endif


class TPgSQLRow : public TSQLRow {

private:
   PGresult *fResult;       // current result set
   ULong_t   fRowNum;       // row number

   Bool_t  IsValid(Int_t field);

public:
   TPgSQLRow(void *result, ULong_t rowHandle);
   ~TPgSQLRow();

   void        Close(Option_t *opt="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);

   ClassDef(TPgSQLRow,0)  // One row of PgSQL query result
};

#endif
