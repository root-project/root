// @(#)root/oracle:$Name:  $:$Id: TOracleStatement.cxx,v 1.3 2006/06/02 14:02:03 brun Exp $
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

// Check that statement is ready for use
#define CheckStatement(method, res)                     \
   {                                                    \
      ClearError();                                     \
      if (fStmt==0) {                                   \
         SetError(-1,"Statement is not correctly initialized",method); \
         return res;                                    \
      }                                                 \
   }

// Check that parameter can be set for statement
#define CheckSetPar(method)                             \
   {                                                    \
      CheckStatement(method, kFALSE);                   \
      if (!IsParSettMode()) {                           \
         SetError(-1,"Parameters cannot be set for this statement", method); \
         return kFALSE;                                 \
      }                                                 \
      if (npar<0) {                                     \
         TString errmsg("Invalid parameter number ");   \
         errmsg+= npar;                                 \
         SetError(-1,errmsg.Data(),method);             \
         return kFALSE;                                 \
      }                                                 \
   }

#define CheckGetField(method)                             \
   {                                                    \
      ClearError();                                     \
      if (!IsResultSet()) {                             \
         SetError(-1,"There is no result set for statement", method); \
         return 0;                                      \
      }                                                 \
      if ((npar<0) || (npar>=fBufferSize)) {                                     \
         TString errmsg("Invalid parameter number ");   \
         errmsg+= npar;                                 \
         SetError(-1,errmsg.Data(),method);             \
         return 0;                                      \
      }                                                 \
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
   
   CheckStatement("Process", kFALSE);

   try {

      if (IsParSettMode()) {
         fStmt->executeUpdate();
         fWorkingMode = 0;
      } else {
         fStmt->execute();
      }

      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Process");
   }

   return kFALSE;
}

//______________________________________________________________________________
Int_t TOracleStatement::GetNumAffectedRows()
{
   // Return number of affected rows after statement Process() was called
   // Make sense for queries like SELECT, INSERT, UPDATE
    
   CheckStatement("GetNumAffectedRows", -1);

   try {
      return fStmt->getUpdateCount();
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetNumAffectedRows");
   }
   return -1;
}


//______________________________________________________________________________
Int_t TOracleStatement::GetNumParameters()
{
   // Return number of parameters in statement
   // Not yet implemented for Oracle 
    
   CheckStatement("GetNumParameters", -1);

   Info("GetParametersNumber","Not implemented");

   return 0;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetNull(Int_t npar)
{
   // Set NULL as value of parameter npar
   
   CheckSetPar("SetNull");

   try {
      fStmt->setNull(npar+1, OCCIINT);

      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetNull");
   }

   return kFALSE;
}


//______________________________________________________________________________
Bool_t TOracleStatement::SetInt(Int_t npar, Int_t value)
{
   // Set integer value for parameter npar
    
   CheckSetPar("SetInt");

   try {
      fStmt->setInt(npar+1, value);

      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetInt");
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetUInt(Int_t npar, UInt_t value)
{
   // Set unsigned integer value for parameter npar

   CheckSetPar("SetUInt");

   try {
      fStmt->setUInt(npar+1, value);
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetUInt");
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetLong(Int_t npar, Long_t value)
{
   // Set long integer value for parameter npar

   CheckSetPar("SetLong");

   try {
      fStmt->setNumber(npar+1, Number(value));
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetLong");
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetLong64(Int_t npar, Long64_t value)
{
   // Set 64-bit integer value for parameter npar

   CheckSetPar("SetLong64");
   
   try {
      fStmt->setNumber(npar+1, Number((long double)value));
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetLong64");
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetULong64(Int_t npar, ULong64_t value)
{
   // Set unsigned 64-bit integer value for parameter npar

   CheckSetPar("SetULong64");

   try {
      fStmt->setNumber(npar+1, Number((long double)value));
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetULong64");
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetDouble(Int_t npar, Double_t value)
{
   // Set double value for parameter npar

   CheckSetPar("SetDouble");
   
   try {
      fStmt->setDouble(npar+1, value);
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetDouble");
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::SetString(Int_t npar, const char* value, Int_t maxsize)
{
   // Set string value for parameter npar

   CheckSetPar("SetString");

   try {

   // this is when NextIteration is called first time
      if (fIterCounter==1) {
         fStmt->setDatabaseNCHARParam(npar+1, true);
         fStmt->setMaxParamSize(npar+1, maxsize);
      }

      fStmt->setString(npar+1, value);
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "SetString");
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::NextIteration()
{
   // Add next iteration for statement with parameters

   CheckStatement("NextIteration", kFALSE);

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
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "NextIteration");
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::StoreResult()
{
   // Store result of statement processing.
   // Required to access results of SELECT queries 

   CheckStatement("StoreResult", kFALSE);

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
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "StoreResult");
   }
   return kFALSE;
}

//______________________________________________________________________________
Int_t TOracleStatement::GetNumFields()
{
   // Returns number of fields in result set 
    
   return IsResultSet() ?  fBufferSize : -1;
}

//______________________________________________________________________________
const char* TOracleStatement::GetFieldName(Int_t npar)
{
   // Return field name in result set 
    
   CheckGetField("GetFieldName");

   if (!IsResultSet() || (npar<0) || (npar>=fBufferSize)) return 0;

   if (fBuffer[npar].namebuf!=0) return fBuffer[npar].namebuf;

   std::string buff = (*fFieldInfo)[npar].getString(MetaData::ATTR_NAME);

   if (buff.length()==0) return 0;

   fBuffer[npar].namebuf = new char[buff.length()+1];

   strcpy(fBuffer[npar].namebuf, buff.c_str());

   return fBuffer[npar].namebuf;
}

//______________________________________________________________________________
Bool_t TOracleStatement::NextResultRow()
{
   // Move cursor to next row in result set.
   // For Oracle it may lead to additional request to database 
    
   ClearError();
   
   if (fResult==0) {
      SetError(-1,"There is no result set for statement", "NextResultRow");
      return kFALSE;
   }

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
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "NextResultRow");
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleStatement::IsNull(Int_t npar)
{
   // Checks if fieled value in result set is NULL  
    
   CheckGetField("IsNull");

   try {
      return fResult->isNull(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "IsNull");
   }

   return kTRUE;
}

//______________________________________________________________________________
Int_t TOracleStatement::GetInt(Int_t npar)
{
   // return field value as integer
    
   CheckGetField("GetInt");

   Int_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = fResult->getInt(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetInt");
   }

   return res;
}

//______________________________________________________________________________
UInt_t TOracleStatement::GetUInt(Int_t npar)
{
   // return field value as unsigned integer

   CheckGetField("GetUInt");

   UInt_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = fResult->getUInt(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetUInt");
   }

   return res;
}


//______________________________________________________________________________
Long_t TOracleStatement::GetLong(Int_t npar)
{
   // return field value as long integer

   CheckGetField("GetLong");

   Long_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = (Long_t) fResult->getNumber(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetLong");
   }

   return res;
}

//______________________________________________________________________________
Long64_t TOracleStatement::GetLong64(Int_t npar)
{
   // return field value as 64-bit integer

   CheckGetField("GetLong64");

   Long64_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = (Long64_t) (long double) fResult->getNumber(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetLong64");
   }

   return res;
}

//______________________________________________________________________________
ULong64_t TOracleStatement::GetULong64(Int_t npar)
{
   // return field value as unsigned 64-bit integer

   CheckGetField("GetULong64");

   ULong64_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = (ULong64_t) (long double) fResult->getNumber(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetULong64");
   }

   return res;
}

//______________________________________________________________________________
Double_t TOracleStatement::GetDouble(Int_t npar)
{
   // return field value as double

   CheckGetField("GetDouble");

   Double_t res = 0;

   try {
      if (!fResult->isNull(npar+1))
        res = fResult->getDouble(npar+1);
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetDouble");
   }

   return res;
}

//______________________________________________________________________________
const char* TOracleStatement::GetString(Int_t npar)
{
   // return field value as string

   CheckGetField("GetString");

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
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "GetString");
   }

   return 0;
}
