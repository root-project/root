// @(#)root/physics:$Id$
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOracleRow
#define ROOT_TOracleRow

#ifndef ROOT_TSQLRow
#include "TSQLRow.h"
#endif

#if !defined(__CINT__)
#ifndef R__WIN32
#include <sys/time.h>
#endif
#include <occi.h>
using namespace oracle::occi;
#ifdef CONST
#undef CONST
#endif
#else
class ResultSet;
class MetaData;
#endif

class TOracleRow : public TSQLRow {

private:
   ResultSet                *fResult;      // current result set
   std::vector<MetaData>    *fFieldInfo;   // metadata for columns
   Int_t                     fFieldCount;
   char                    **fFieldsBuffer;

   Bool_t  IsValid(Int_t field);

protected:
   void        GetRowData();

public:
   TOracleRow(ResultSet *rs, std::vector<MetaData> *fieldMetaData);
   ~TOracleRow();

   void        Close(Option_t *opt="");
   ULong_t     GetFieldLength(Int_t field);
   const char *GetField(Int_t field);

   ClassDef(TOracleRow,0)  // One row of Oracle query result
};

#endif
