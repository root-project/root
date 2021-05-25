// @(#)root/odbc:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TODBCResult
#define ROOT_TODBCResult

#include "TSQLResult.h"

#include "TString.h"


#ifdef __CLING__
typedef void * SQLHSTMT;
#else
#ifdef WIN32
#include "windows.h"
#endif
#include <sql.h>
#endif


class TODBCResult : public TSQLResult {

protected:
   SQLHSTMT    fHstmt;
   Int_t       fFieldCount{0};
   TString     fNameBuffer;

public:
   TODBCResult(SQLHSTMT stmt);
   virtual ~TODBCResult();

   void        Close(Option_t *opt="") final;
   Int_t       GetFieldCount() final { return fFieldCount; }
   const char *GetFieldName(Int_t field) final;
   TSQLRow    *Next() final;

   ClassDefOverride(TODBCResult,0)  // ODBC query result
};

#endif
