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

#ifndef ROOT_TSQLRow
#include "TSQLRow.h"
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

class TODBCRow : public TSQLRow {

protected:
   SQLHSTMT   fHstmt;
   Int_t      fFieldCount;
   char      **fBuffer;
   ULong_t    *fLengths;  
   
   void        CopyFieldValue(Int_t field);
   
private:
   TODBCRow(const TODBCRow&);            // Not implemented.
   TODBCRow &operator=(const TODBCRow&); // Not implemented.

public:
   TODBCRow(SQLHSTMT stmt, Int_t fieldcount);
   virtual ~TODBCRow();

   void        Close(Option_t *opt="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);

   ClassDef(TODBCRow,0)  // One row of ODBC query result
};

#endif
