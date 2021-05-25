// @(#)root/odbc:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TODBCStatement
#define ROOT_TODBCStatement

#include "TSQLStatement.h"


#ifdef __CLING__
typedef void *   SQLHSTMT;
typedef UShort_t SQLUSMALLINT;
typedef UInt_t   SQLUINTEGER;
typedef Short_t  SQLSMALLINT;
typedef Short_t  SQLRETURN;
#else
#ifdef WIN32
#include "windows.h"
#endif
#include <sql.h>
#endif

class TODBCStatement : public TSQLStatement {

protected:
    #ifdef __CLING__
    struct ODBCBufferRec_t;
    #else
    struct ODBCBufferRec_t {
       Int_t       fBroottype;
       Int_t       fBsqltype;
       Int_t       fBsqlctype;
       void       *fBbuffer;
       Int_t       fBelementsize;
       SQLLEN     *fBlenarray;
       char       *fBstrbuffer;
       char       *fBnamebuffer;
    };
    #endif

protected:
   SQLHSTMT         fHstmt;
   Int_t            fBufferPreferredSize{0};
   ODBCBufferRec_t *fBuffer{nullptr};
   Int_t            fNumBuffers{0};
   Int_t            fBufferLength{0};     // number of entries for each parameter/column
   Int_t            fBufferCounter{0};    // used to indicate position in buffers
   SQLUSMALLINT    *fStatusBuffer{nullptr};
   Int_t            fWorkingMode{0};      // 1 - setting parameters, 2 - reading results, 0 - unknown
   SQLUINTEGER      fNumParsProcessed{0};  // contains number of parameters, affected by last operation
   SQLUINTEGER      fNumRowsFetched{0};    // indicates number of fetched rows
   ULong64_t        fLastResultRow{0};     // stores values of row number after last fetch operation

   void       *GetParAddr(Int_t npar, Int_t roottype = 0, Int_t length = 0);
   long double ConvertToNumeric(Int_t npar);
   const char *ConvertToString(Int_t npar);

   Bool_t      BindColumn(Int_t ncol, SQLSMALLINT sqltype, SQLUINTEGER size);
   Bool_t      BindParam(Int_t n, Int_t type, Int_t size = 1024);

   Bool_t      ExtractErrors(SQLRETURN retcode, const char* method);

   void        SetNumBuffers(Int_t isize, Int_t ilen);
   void        FreeBuffers();

   Bool_t      IsParSettMode() const { return fWorkingMode==1; }
   Bool_t      IsResultSet() const { return fWorkingMode==2; }

public:
   TODBCStatement(SQLHSTMT stmt, Int_t rowarrsize, Bool_t errout = kTRUE);
   virtual ~TODBCStatement();

   virtual void        Close(Option_t * = "") final;

   Int_t       GetBufferLength() const final { return fBufferLength; }
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

   Bool_t      IsNull(Int_t) final;
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

   ClassDefOverride(TODBCStatement, 0); //ODBC implementation of TSQLStatement
};

#endif
