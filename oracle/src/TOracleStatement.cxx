// @(#)root/oracle:$Name:  $:$Id: TOracleStatement.cxx,v 1.1 2006/02/6 10:00:44 rdm Exp $
// Author: Sergey Linev   6/02/2006


/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  SQL statement class for Oracle                                      //
//                                                                      //
//  See TSQLStatement class documentation for more details.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TOracleStatement.h"
#include "TDataType.h"

ClassImp(TOracleStatement)

//______________________________________________________________________________
TOracleStatement::TOracleStatement(Connection* conn, Statement* stmt, Int_t niter) :
   TSQLStatement(),
   fConn(conn),
   fStmt(stmt),
   fResult(0),
   fFieldInfo(0),
   fBuffer(0),
   fBufferSize(0),
   fNumIterations(niter),
   fIterCounter(0),
   fWorkingMode(0)
{
   // Normal constructor of TOracleStatement class
   // On creation time specifies buffer length, which should be
   // used in data fetching or data inserting

   if (fStmt) {
      fStmt->setPrefetchMemorySize(1000000);
      fStmt->setPrefetchRowCount(niter);
      fStmt->setMaxIterations(niter);
   }
}

//______________________________________________________________________________
TOracleStatement::~TOracleStatement()
{
   // Destructor of TOracleStatement clas

   Close();
}

//______________________________________________________________________________
void TOracleStatement::Close(Option_t *)
{
   // Close Oracle statement
   // Removes and destroys all buffers and metainfo


   if (fFieldInfo)
      delete fFieldInfo;

   if (fResult && fStmt)
      fStmt->closeResultSet(fResult);

   if (fConn && fStmt)
      fConn->terminateStatement(fStmt);

   CloseBuffer();

   fConn = 0;
   fStmt = 0;
   fResult = 0;
   fFieldInfo = 0;
   fIterCounter = 0;
}

//______________________________________________________________________________
void TOracleStatement::SetBufferSize(Int_t size)
{
    // Set buffer size, which is used to keep string values of
    // currently fetched column.


    CloseBuffer();
    if (size<=0) return;
    fBufferSize = size;
    fBuffer = new TBufferRec[size];
    for (Int_t n=0;n<fBufferSize;n++) {
       fBuffer[n].strbuf = 0;
       fBuffer[n].namebuf = 0;
    }
}

//______________________________________________________________________________
void TOracleStatement::CloseBuffer()
{
   // Destroy buffers, used in data fetching

   if (fBuffer) {
      for (Int_t n=0;n<fBufferSize;n++) {
         delete[] fBuffer[n].strbuf;
         delete[] fBuffer[n].namebuf;
      }

      delete[] fBuffer;
   }
   fBuffer = 0;
   fBufferSize = 0;
}

//______________________________________________________________________________
Bool_t TOracleStatement::Process()
{
   // Process SQL statement

   if (fStmt==0) return kFALSE;

   try {

      if (IsParSettMode()) {
         fStmt->executeUpdate();
         fWorkingMode = 0;
      } else {
         fStmt->execute();
      }

      return kTRUE;
   } catch (SQLException &oraex)  {
      Error("Process", "\nsql:%s\nerr:%s", fStmt->getSQL().c_str(), (oraex.getMessage()).c_str());
   }

   return kFALSE;
}

//______________________________________________________________________________
Int_t TOracleStatement::GetNumAffectedRows()
{
   if (fStmt==0) return -1;

   try {
      return fStmt->getUpdateCount();
   } catch (SQLException &oraex)  {
      Error("GetNumAffectedRows", oraex.getMessage().c_str());
   }
   return -1;
}


//______________________________________________________________________________
Int_t TOracleStatement::GetNumParameters()
{
   if (fStmt==0) return -1;

   Info("GetParametersNumber","Not implemented");

   return 0;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetInt(Int_t npar, Int_t value)
{
   if (!IsParSettMode()) return kFALSE;

   try {
      fStmt->setInt(npar+1, value);

      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setInt", (oraex.getMessage()).c_str());
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetUInt(Int_t npar, UInt_t value)
{
   if (!IsParSettMode()) return kFALSE;

   try {
      fStmt->setUInt(npar+1, value);
      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setUInt", (oraex.getMessage()).c_str());
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetLong(Int_t npar, Long_t value)
{
   if (!IsParSettMode()) return kFALSE;
   try {
      fStmt->setNumber(npar+1, Number(value));
      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setLong", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetLong64(Int_t npar, Long64_t value)
{
   if (!IsParSettMode()) return kFALSE;
   try {
      fStmt->setNumber(npar+1, Number((long double)value));
      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setLong64", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetULong64(Int_t npar, ULong64_t value)
{
   if (!IsParSettMode()) return kFALSE;
   try {
      fStmt->setNumber(npar+1, Number((long double)value));
      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setULong64", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetDouble(Int_t npar, Double_t value)
{
   if (!IsParSettMode()) return kFALSE;
   try {
      fStmt->setDouble(npar+1, value);
      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setDouble", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetString(Int_t npar, const char* value, Int_t maxsize)
{
   if (!IsParSettMode()) return kFALSE;

   try {

   // this is when NextIteration is called first time
      if (fIterCounter==1) {
         fStmt->setDatabaseNCHARParam(npar+1, true);
         fStmt->setMaxParamSize(npar+1, maxsize);
      }

      fStmt->setString(npar+1, value);
      return kTRUE;
   } catch (SQLException &oraex)  {
       Error("setLong64", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::NextIteration()
{
   if (fStmt==0) return kFALSE;

   try {
      fWorkingMode=1;
      // if number of iterations achievs limit, execute it and continue to fill
      if ((fIterCounter % fNumIterations == 0) && (fIterCounter>0)) {
         fStmt->executeUpdate();
      }

      if (fIterCounter % fNumIterations != 0) {
         fStmt->addIteration();
      }

      fIterCounter++;

      return kTRUE;
   } catch (SQLException &oraex)  {
      Error("NextIteration", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::StoreResult()
{

   if (fStmt==0) return kFALSE;

   try {
      if (fStmt->status() == Statement::RESULT_SET_AVAILABLE) {
         fResult      = fStmt->getResultSet();
         fFieldInfo   = (fResult==0) ? 0 : new std::vector<MetaData>(fResult->getColumnListMetaData());
         Int_t count  = (fFieldInfo==0) ? 0 : fFieldInfo->size();
         SetBufferSize(count);
         if ((fResult!=0) && (count>0)) fWorkingMode = 2;

         return IsResultSet();
      }
   } catch (SQLException &oraex) {
      Error("StoreResult()", (oraex.getMessage()).c_str());
   }
   return kFALSE;
}

//______________________________________________________________________________
Int_t TOracleStatement::GetNumFields()
{
   return IsResultSet() ?  fBufferSize : -1;
}

//______________________________________________________________________________
const char* TOracleStatement::GetFieldName(Int_t nfield)
{
   if (!IsResultSet() || (nfield<0) || (nfield>=fBufferSize)) return 0;

   if (fBuffer[nfield].namebuf!=0) return fBuffer[nfield].namebuf;

   std::string buff = (*fFieldInfo)[nfield].getString(MetaData::ATTR_NAME);

   if (buff.length()==0) return 0;

   fBuffer[nfield].namebuf = new char[buff.length()+1];

   strcpy(fBuffer[nfield].namebuf, buff.c_str());

   return fBuffer[nfield].namebuf;
}


//______________________________________________________________________________
Bool_t TOracleStatement::NextResultRow()
{
   if (fResult==0) return kFALSE;

   try {
      for (int n=0;n<fBufferSize;n++)
        if (fBuffer[n].strbuf) {
           delete[] fBuffer[n].strbuf;
           fBuffer[n].strbuf = 0;
        }
      if (!fResult->next()) {
         fWorkingMode = 0;
         CloseBuffer();
         return kFALSE;
      }
      return kTRUE;
   } catch (SQLException &oraex) {
      Error("NextResultRow", (oraex.getMessage()).c_str());
   }

   return kFALSE;
}

//______________________________________________________________________________
Int_t TOracleStatement::GetInt(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   Int_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = fResult->getInt(npar+1);
   } catch (SQLException &oraex) {
      Error("getInt", (oraex.getMessage()).c_str());
   }

   return res;
}

//______________________________________________________________________________
UInt_t TOracleStatement::GetUInt(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   UInt_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = fResult->getUInt(npar+1);
   } catch (SQLException &oraex) {
      Error("getUInt", (oraex.getMessage()).c_str());
   }

   return res;
}


//______________________________________________________________________________
Long_t TOracleStatement::GetLong(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   Long_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = (Long_t) fResult->getNumber(npar+1);
   } catch (SQLException &oraex) {
      Error("getLong", (oraex.getMessage()).c_str());
   }

   return res;
}

//______________________________________________________________________________
Long64_t TOracleStatement::GetLong64(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   Long64_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = (Long64_t) (long double) fResult->getNumber(npar+1);
   } catch (SQLException &oraex) {
      Error("getLong64", (oraex.getMessage()).c_str());
   }

   return res;
}

//______________________________________________________________________________
ULong64_t TOracleStatement::GetULong64(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   ULong64_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = (ULong64_t) (long double) fResult->getNumber(npar+1);
   } catch (SQLException &oraex) {
      Error("getULong64", (oraex.getMessage()).c_str());
   }

   return res;
}

//______________________________________________________________________________
Double_t TOracleStatement::GetDouble(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   Double_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = fResult->getDouble(npar+1);
   } catch (SQLException &oraex) {
      Error("getDouble", (oraex.getMessage()).c_str());
   }

   return res;
}

//______________________________________________________________________________
const char* TOracleStatement::GetString(Int_t npar)
{
   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   if (fBuffer[npar].strbuf!=0) return fBuffer[npar].strbuf;

   try {
      if (fResult->isNull(npar+1)) return 0;

      int datatype = (*fFieldInfo)[npar].getInt(MetaData::ATTR_DATA_TYPE);

      std::string res;

      switch (datatype) {
        case 2: { //NUMBER
           int prec = (*fFieldInfo)[npar].getInt(MetaData::ATTR_PRECISION);
           int scale = (*fFieldInfo)[npar].getInt(MetaData::ATTR_SCALE);

           if ((scale == 0) || (prec == 0)) {
              res = fResult->getString(npar+1);
           } else {
              double double_val = fResult->getDouble(npar+1);
              char str_number[50];
              sprintf(str_number, "%lf", double_val);
              res = str_number;
           }
           break;
        }
        case 1:  // VARCHAR2
        case 12: // DATE
        case 96:  // CHAR
           res = fResult->getString(npar+1);
           break;
        case 187: // TIMESTAMP
        case 188: // TIMESTAMP WITH TIMEZONE
        case 232: // TIMESTAMP WITH LOCAL TIMEZONE
           res = (fResult->getTimestamp(npar+1)).toText("MM/DD/YYYY, HH24:MI:SS",0);
           break;
        default:
           res = fResult->getString(npar+1);
           Info("getString","Type %d may not be supported");
      }

      int len = res.length();

      if (len>0) {
          fBuffer[npar].strbuf = new char[len+1];
          strcpy(fBuffer[npar].strbuf, res.c_str());
      }

      return fBuffer[npar].strbuf;

   } catch (SQLException &oraex) {
      Error("getString", (oraex.getMessage()).c_str());
   }

   return 0;
}
