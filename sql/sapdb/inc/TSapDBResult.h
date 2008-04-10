// @(#)root/sapdb:$Id$
// Author: Mark Hemberger & Fons Rademakers   03/08/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSapDBResult
#define ROOT_TSapDBResult

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

#if !defined(__CINT__)
#include <sys/time.h>
#include <WINDOWS.H>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include <sql.h>
#include <sqlext.h>
#else
typedef long int SDWORD;
typedef long     SQLHSTMT;
#endif

class TString;


class TSapDBResult : public TSQLResult {

private:
   SQLHSTMT  fResult;      // query result (rows)
   TString  *fFieldNames;  // names of fields
   Int_t     fFieldCount;  // number of fields

   Bool_t  IsValid(Int_t field);

public:
   TSapDBResult(SQLHSTMT fStmt, SDWORD rowCount = 0);
   ~TSapDBResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TSapDBResult,0)  // SapDB query result
};

#endif
