// @(#)root/mysql:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMySQLStatement
#define ROOT_TMySQLStatement

#include "TSQLStatement.h"


#include <mysql.h>

#if MYSQL_VERSION_ID < 40100
typedef struct { int dummy; } MYSQL_STMT;
typedef struct { int dummy; } MYSQL_BIND;
#endif

// MariaDB is fork of MySQL and still include definition of my_bool
// MariaDB major version is 10, therefore it confuses version ID here
#ifndef MARIADB_VERSION_ID
#if MYSQL_VERSION_ID > 80000 && MYSQL_VERSION_ID < 100000
typedef bool my_bool;
#endif
#endif

class TMySQLStatement : public TSQLStatement {

protected:

   struct TParamData {
      void         *fMem{nullptr};        //! allocated data buffer
      Int_t         fSize{0};             //! size of allocated data
      Int_t         fSqlType{0};          //! sqltype of parameter
      Bool_t        fSign{kFALSE};        //! signed - not signed type
      ULong_t       fResLength{0};        //! length argument
      my_bool       fResNull{false};      //! indicates if argument is null
      char         *fStrBuffer{nullptr};  //! special buffer to be used for string conversions
      std::string   fFieldName;           //! buffer for field name
   };

   MYSQL_STMT      *fStmt{nullptr};       //! executed statement
   Int_t            fNumBuffers{0};       //! number of statement parameters
   MYSQL_BIND      *fBind{nullptr};       //! array of bind data
   TParamData      *fBuffer{nullptr};     //! parameter definition structures
   Int_t            fWorkingMode{0};      //! 1 - setting parameters, 2 - retrieving results
   Int_t            fIterationCount{-1};  //! number of iteration
   Bool_t           fNeedParBind{kFALSE}; //! indicates when parameters bind should be called

   Bool_t      IsSetParsMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSetMode() const { return fWorkingMode==2; }

   Bool_t      SetSQLParamType(Int_t npar, int sqltype, Bool_t sig, ULong_t sqlsize = 0);

   long double ConvertToNumeric(Int_t npar);
   const char *ConvertToString(Int_t npar);

   void        FreeBuffers();
   void        SetBuffersNumber(Int_t n);

   void       *BeforeSet(const char* method, Int_t npar, Int_t sqltype, Bool_t sig = kTRUE, ULong_t size = 0);

   static ULong64_t fgAllocSizeLimit;

private:
   TMySQLStatement(const TMySQLStatement&) = delete;
   TMySQLStatement &operator=(const TMySQLStatement&) = delete;

public:
   TMySQLStatement(MYSQL_STMT* stmt, Bool_t errout = kTRUE);
   virtual ~TMySQLStatement();

   static ULong_t GetAllocSizeLimit() { return fgAllocSizeLimit; }
   static void SetAllocSizeLimit(ULong_t sz) { fgAllocSizeLimit = sz; }

   void        Close(Option_t * = "") final;

   Int_t       GetBufferLength() const final { return 1; }
   Int_t       GetNumParameters() final;

   Bool_t      SetNull(Int_t npar) final;
   Bool_t      SetInt(Int_t npar, Int_t value) final;
   Bool_t      SetUInt(Int_t npar, UInt_t value) final;
   Bool_t      SetLong(Int_t npar, Long_t value) final;
   Bool_t      SetLong64(Int_t npar, Long64_t value) final;
   Bool_t      SetULong64(Int_t npar, ULong64_t value) final;
   Bool_t      SetDouble(Int_t npar, Double_t value) final;
   Bool_t      SetString(Int_t npar, const char* value, Int_t maxsize = 256) final;
   Bool_t      SetBinary(Int_t npar, void* mem, Long_t size, Long_t maxsize = 0x1000) final;
   Bool_t      SetDate(Int_t npar, Int_t year, Int_t month, Int_t day) final;
   Bool_t      SetTime(Int_t npar, Int_t hour, Int_t min, Int_t sec) final;
   Bool_t      SetDatime(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec) final;
   using TSQLStatement::SetTimestamp;
   Bool_t      SetTimestamp(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec, Int_t frac = 0) final;

   Bool_t      NextIteration() final;

   Bool_t      Process() final;
   Int_t       GetNumAffectedRows() final;

   Bool_t      StoreResult() final;
   Int_t       GetNumFields() final;
   const char *GetFieldName(Int_t nfield) final;
   Bool_t      NextResultRow() final;

   Bool_t      IsNull(Int_t npar) final;
   Int_t       GetInt(Int_t npar) final;
   UInt_t      GetUInt(Int_t npar) final;
   Long_t      GetLong(Int_t npar) final;
   Long64_t    GetLong64(Int_t npar) final;
   ULong64_t   GetULong64(Int_t npar) final;
   Double_t    GetDouble(Int_t npar) final;
   const char *GetString(Int_t npar) final;
   Bool_t      GetBinary(Int_t npar, void* &mem, Long_t& size) final;
   Bool_t      GetDate(Int_t npar, Int_t& year, Int_t& month, Int_t& day) final;
   Bool_t      GetTime(Int_t npar, Int_t& hour, Int_t& min, Int_t& sec) final;
   Bool_t      GetDatime(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec) final;
   using TSQLStatement::GetTimestamp;
   Bool_t      GetTimestamp(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec, Int_t&) final;

   ClassDefOverride(TMySQLStatement, 0);  // SQL statement class for MySQL DB
};

#endif
