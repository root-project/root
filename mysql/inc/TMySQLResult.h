// @(#)root/mysql:$Name:  $:$Id: TMySQLResult.h,v 1.1.1.1 2000/05/16 17:00:58 rdm Exp $
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMySQLResult
#define ROOT_TMySQLResult

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

#if !defined(__CINT__)
#ifndef R__WIN32
#include <sys/time.h>
#endif
#include <mysql.h>
#else
struct MYSQL_RES;
struct MYSQL_FIELD;
#endif


class TMySQLResult : public TSQLResult {

private:
   MYSQL_RES   *fResult;      // query result (rows)
   MYSQL_FIELD *fFieldInfo;   // info for each field in the row

   Bool_t  IsValid(Int_t field);

public:
   TMySQLResult(void *result);
   ~TMySQLResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TMySQLResult,0)  // MySQL query result
};

#endif
