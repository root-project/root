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

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


#ifdef __CINT__
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
   Int_t       fFieldCount;
   TString     fNameBuffer;

public:
   TODBCResult(SQLHSTMT stmt);
   virtual ~TODBCResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount() { return fFieldCount; }
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TODBCResult,0)  // ODBC query result
};

#endif
