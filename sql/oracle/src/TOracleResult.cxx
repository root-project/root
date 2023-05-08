// @(#)root/oracle:$Id$
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TOracleResult.h"
#include "TOracleRow.h"
#include "TList.h"

#ifndef R__WIN32
#include <sys/time.h>
#endif

#include <occi.h>

ClassImp(TOracleResult);

////////////////////////////////////////////////////////////////////////////////
/// Oracle query result.

void TOracleResult::initResultSet(oracle::occi::Statement *stmt)
{
   if (!stmt) {
      Error("initResultSet", "construction: empty statement");
   } else {
      try {
         fStmt = stmt;
         if (stmt->status() == oracle::occi::Statement::RESULT_SET_AVAILABLE) {
            fResultType  = 1;
            fResult      = stmt->getResultSet();
            fFieldInfo   = fResult ? new std::vector<oracle::occi::MetaData>(fResult->getColumnListMetaData()) : nullptr;
            fFieldCount  = fFieldInfo ? fFieldInfo->size() : 0;
         } else if (stmt->status() == oracle::occi::Statement::UPDATE_COUNT_AVAILABLE) {
            fResultType  = 3; // this is update_count_available
            fResult      = nullptr;
            fFieldInfo   = nullptr;
            fFieldCount  = 0;
            fUpdateCount = stmt->getUpdateCount();
         }
      } catch (oracle::occi::SQLException &oraex) {
         Error("initResultSet", "%s", (oraex.getMessage()).c_str());
         MakeZombie();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////

TOracleResult::TOracleResult(oracle::occi::Connection *conn, oracle::occi::Statement *stmt)
{
   fConn        = conn;
   fResult      = nullptr;
   fStmt        = nullptr;
   fPool        = nullptr;
   fRowCount    = 0;
   fFieldInfo   = nullptr;
   fResultType  = 0;
   fUpdateCount = 0;

   initResultSet(stmt);

   if (fResult) ProducePool();
}

////////////////////////////////////////////////////////////////////////////////
/// This construction func is only used to get table metainfo.

TOracleResult::TOracleResult(oracle::occi::Connection *conn, const char *tableName)
{
   fResult      = nullptr;
   fStmt        = nullptr;
   fConn        = nullptr;
   fPool        = nullptr;
   fRowCount    = 0;
   fFieldInfo   = nullptr;
   fResultType  = 0;
   fUpdateCount = 0;
   fFieldCount  = 0;

   if (!tableName || !conn) {
      Error("TOracleResult", "construction: empty input parameter");
   } else {
      oracle::occi::MetaData connMD = conn->getMetaData(tableName, oracle::occi::MetaData::PTYPE_TABLE);
      fFieldInfo   = new std::vector<oracle::occi::MetaData>(connMD.getVector(oracle::occi::MetaData::ATTR_LIST_COLUMNS));
      fFieldCount  = fFieldInfo->size();
      fResultType  = 2; // indicates that this is just an table metainfo
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup Oracle query result.

TOracleResult::~TOracleResult()
{
   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close query result.

void TOracleResult::Close(Option_t *)
{
   if (fConn && fStmt) {
      if (fResult) fStmt->closeResultSet(fResult);
      fConn->terminateStatement(fStmt);
   }

   if (fPool) {
      fPool->Delete();
      delete fPool;
   }

   if (fFieldInfo)
      delete fFieldInfo;

   fResultType = 0;

   fStmt = nullptr;
   fResult = nullptr;
   fFieldInfo = nullptr;
   fPool = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if result set is open and field index within range.

Bool_t TOracleResult::IsValid(Int_t field)
{
   if (field < 0 || field >= fFieldCount) {
      Error("IsValid", "field index out of bounds");
      return kFALSE;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get number of fields in result.

Int_t TOracleResult::GetFieldCount()
{
   return fFieldCount;
}

////////////////////////////////////////////////////////////////////////////////
/// Get name of specified field.

const char *TOracleResult::GetFieldName(Int_t field)
{
   if (!IsValid(field))
      return nullptr;
   fNameBuffer = (*fFieldInfo)[field].getString(oracle::occi::MetaData::ATTR_NAME);
   return fNameBuffer.c_str();
}

////////////////////////////////////////////////////////////////////////////////
/// Get next query result row. The returned object must be
/// deleted by the user.

TSQLRow *TOracleResult::Next()
{
   if (!fResult || (fResultType!=1)) return nullptr;

   if (fPool) {
      TSQLRow* row = (TSQLRow*) fPool->First();
      if (row) fPool->Remove(row);
      return row;
   }

   // if select query,
   try {
      if (fResult->next() != oracle::occi::ResultSet::END_OF_FETCH) {
         fRowCount++;
         return new TOracleRow(fResult, fFieldInfo);
      } else
         return nullptr;
   } catch (oracle::occi::SQLException &oraex) {
      Error("Next", "%s", (oraex.getMessage()).c_str());
      MakeZombie();
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Int_t TOracleResult::GetRowCount() const
{
   if (!fResult) return 0;

   if (!fPool) ((TOracleResult*) this)->ProducePool();

   return fRowCount;
}

////////////////////////////////////////////////////////////////////////////////

void TOracleResult::ProducePool()
{
   if (fPool) return;

   TList* pool = new TList;
   TSQLRow* res = nullptr;
   while ((res = Next()) != nullptr) {
      pool->Add(res);
   }

   fPool = pool;
}
