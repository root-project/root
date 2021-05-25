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

#include "TSQLRow.h"

#include <mysql.h>

class TMySQLRow : public TSQLRow {

private:
   MYSQL_RES   *fResult{nullptr};      // current result set
   MYSQL_ROW    fFields;               // current row
   ULong_t     *fFieldLength{nullptr}; // length of each field in the row

   Bool_t  IsValid(Int_t field);

public:
   TMySQLRow(void *result, ULong_t rowHandle);
   ~TMySQLRow();

   void        Close(Option_t *opt="") final;
   ULong_t     GetFieldLength(Int_t field) final;
   const char *GetField(Int_t field) final;

   ClassDefOverride(TMySQLRow,0)  // One row of MySQL query result
};

#endif

