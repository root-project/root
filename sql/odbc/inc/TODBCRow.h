// @(#)root/odbc:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TODBCRow
#define ROOT_TODBCRow

#include "TSQLRow.h"

#include "TString.h"

#ifdef __CLING__
typedef void * SQLHSTMT;
#else
#ifdef WIN32
#include "windows.h"
#endif
#include <sql.h>
#endif

class TODBCRow : public TSQLRow {

protected:
   SQLHSTMT   fHstmt;
   Int_t      fFieldCount{0};
   char      **fBuffer{nullptr};
   ULong_t    *fLengths{nullptr};

   void        CopyFieldValue(Int_t field);

private:
   TODBCRow(const TODBCRow&) = delete;
   TODBCRow &operator=(const TODBCRow&) = delete;

public:
   TODBCRow(SQLHSTMT stmt, Int_t fieldcount);
   virtual ~TODBCRow();

   void        Close(Option_t *opt="") final;
   ULong_t     GetFieldLength(Int_t field) final;
   const char *GetField(Int_t field) final;

   ClassDefOverride(TODBCRow,0)  // One row of ODBC query result
};

#endif
