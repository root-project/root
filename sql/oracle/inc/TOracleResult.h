// @(#)root/physics:$Id$
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

#include <vector>

#if !defined(__CINT__)
#ifndef R__WIN32
#include <sys/time.h>
#endif
#include <occi.h>
#ifdef CONST
#undef CONST
#endif
#else
namespace oracle { namespace occi {
class Connection;
class Statement;
class ResultSet;
class MetaData;
   }}
#endif

class TList;

class TOracleResult : public TSQLResult {

private:
   oracle::occi::Connection*fConn;        // connection to Oracle 
   oracle::occi::Statement *fStmt;        // executed statement
   oracle::occi::ResultSet *fResult;      // query result (rows)
   std::vector<oracle::occi::MetaData> *fFieldInfo;   // info for each field in the row
   Int_t                  fFieldCount;  // num of fields in resultset
   UInt_t                 fUpdateCount; // for dml query, mutual exclusive with above
   Int_t                  fResultType;  // 0 - nothing; 1 - Select; 2 - table metainfo, 3 - update counter
   TList                 *fPool;        // array of results, produced when number of rows are requested 
   std::string           fNameBuffer; // buffer for GetFieldName() argument

   Bool_t  IsValid(Int_t field);

   TOracleResult(const TOracleResult&);            // Not implemented;
   TOracleResult &operator=(const TOracleResult&); // Not implemented;
   
protected:
   void    initResultSet(oracle::occi::Statement *stmt);
   void    ProducePool();

public:
   TOracleResult(oracle::occi::Connection *conn, oracle::occi::Statement *stmt);
   TOracleResult(oracle::occi::Connection *conn, const char *tableName);
   ~TOracleResult();

   void        Close(Option_t *opt="");
   Int_t       GetFieldCount();
   const char *GetFieldName(Int_t field);
   virtual Int_t GetRowCount() const;
   TSQLRow    *Next();
   
   Int_t       GetUpdateCount() { return fUpdateCount; }

   ClassDef(TOracleResult,0)  // Oracle query result
};

#endif
