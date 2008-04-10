// @(#)root/sapdb:$Id$
// Author: Mark Hemberger & Fons Rademakers   03/08/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSapDBRow
#define ROOT_TSapDBRow

#ifndef ROOT_TSQLRow
#include "TSQLRow.h"
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
typedef long SQLHSTMT;
#endif

class TString;


class TSapDBRow : public TSQLRow {

private:
   SQLHSTMT    fResult;       // current result set
   Int_t       fFieldCount;   // number of fields in row
   ULong_t    *fFieldLength;  // length of each field in the row
   TString    *fFieldValue;   // value of each field in the row

   Bool_t  IsValid(Int_t field);

public:
   TSapDBRow(SQLHSTMT fResult, Int_t nfields);
   ~TSapDBRow();

   void        Close(Option_t *opt="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);

   ClassDef(TSapDBRow,0)  // One row of SapDB query result
};

#endif
