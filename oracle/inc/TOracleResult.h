// @(#)root/physics:$Name: v4-00-08 $:$Id: TOracleResult.h,v 1.0 2004/12/04 17:00:45 rdm Exp $
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOracleResult
#define ROOT_TOracleResult

#ifndef ROOT_TSQLResult
#include "TSQLResult.h"
#endif

#if !defined(__CINT__)
#ifndef R__WIN32
#include <sys/time.h>
#endif
#include <occi.h>
using namespace std;
using namespace oracle::occi;
#else
/*namespace oracle {
   namespace occi { class Statement; }
   namespace occi { class ResultSet; }
   namespace occi { class MetaData;  }
}
namespace std {
   using vector:
}*/
class Statement;
class ResultSet;
class MetaData;
#endif


class TOracleResult : public TSQLResult {

private:
   Statement        *fStmt;   
   ResultSet        *fResult;      // query result (rows)
   vector<MetaData> *fFieldInfo;   // info for each field in the row
   Int_t             fFieldCount;  // num of fields in resultset
   UInt_t            fUpdateCount; //for dml query, mutual exclusive with above
   Int_t             fResultType;  // 0 - Update(dml); 1 - Select; -1 - empty
   
   Bool_t  IsValid(Int_t field);
   void    GetMetaDataInfo();

public:
   TOracleResult(Statement *stmt);
   ~TOracleResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   TSQLRow    *Next();

   ClassDef(TOracleResult,0)  // Oracle query result
};

#endif
