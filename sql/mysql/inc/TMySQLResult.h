// @(#)root/mysql:$Id$
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

#include "TSQLResult.h"

#include <mysql.h>

class TMySQLResult : public TSQLResult {

private:
   MYSQL_RES   *fResult{nullptr};      // query result (rows)
   MYSQL_FIELD *fFieldInfo{nullptr};   // info for each field in the row

   Bool_t  IsValid(Int_t field);

public:
   TMySQLResult(void *result);
   ~TMySQLResult();

   void        Close(Option_t *opt="") final;
   Int_t       GetFieldCount() final;
   const char *GetFieldName(Int_t field) final;
   TSQLRow    *Next() final;

   ClassDefOverride(TMySQLResult,0)  // MySQL query result
};

#endif
