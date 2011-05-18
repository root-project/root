// @(#)root/mysql:$Id$
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMySQLRow
#define ROOT_TMySQLRow

#ifndef ROOT_TSQLRow
#include "TSQLRow.h"
#endif

#if !defined(__CINT__)
#ifdef R__WIN32
#include <winsock2.h>
#else
#include <sys/time.h>
#endif
#include <mysql.h>
#else
struct MYSQL_RES;
typedef char **MYSQL_ROW;
#endif


class TMySQLRow : public TSQLRow {

private:
   MYSQL_RES   *fResult;       // current result set
   MYSQL_ROW    fFields;       // current row
   ULong_t     *fFieldLength;  // length of each field in the row

   Bool_t  IsValid(Int_t field);

public:
   TMySQLRow(void *result, ULong_t rowHandle);
   ~TMySQLRow();

   void        Close(Option_t *opt="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);

   ClassDef(TMySQLRow,0)  // One row of MySQL query result
};

#endif

