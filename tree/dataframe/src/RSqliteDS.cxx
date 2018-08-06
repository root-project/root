// Author: Jakob Blomer CERN  07/2018

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::RDF::RSqliteDS
    \ingroup dataframe
    \brief RDataFrame data source class for reading SQlite files.
*/
// clang-format on

#include <ROOT/RSqliteDS.hxx>
#include <ROOT/RDFUtils.hxx>
#include <ROOT/RMakeUnique.hxx>

#include <TError.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ROOT {

namespace RDF {

RSqliteDS::RSqliteDS(std::string_view fileName, std::string_view query)
   : fDb(nullptr)
   , fQuery(nullptr)
   , fNSlots(0)
   , fNRow(0)
{
   int retval;

   retval = sqlite3_open_v2(std::string(fileName).c_str(), &fDb,
     SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
   if (retval != SQLITE_OK) SqliteError(retval);

   retval = sqlite3_prepare_v2(fDb, std::string(query).c_str(), -1, &fQuery, nullptr);
   if (retval != SQLITE_OK) SqliteError(retval);

   int colCount = sqlite3_column_count(fQuery);
   retval = sqlite3_step(fQuery);
   if ((retval != SQLITE_ROW) && (retval != SQLITE_DONE)) SqliteError(retval);

   fValues.resize(colCount);
   for (int i = 0; i < colCount; ++i) {
      fColumnNames.emplace_back(sqlite3_column_name(fQuery, i));
      int type = SQLITE_NULL;
      // Try first with the declared column type and then with the dynamic type
      // for expressions
      const char *declTypeCstr = sqlite3_column_decltype(fQuery, i);
      if (declTypeCstr == nullptr) {
         if (retval == SQLITE_ROW) type = sqlite3_column_type(fQuery, i);
      } else {
         std::string declType(declTypeCstr);
         if (declType == "INTEGER") type = SQLITE_INTEGER;
         else if (declType == "FLOAT") type = SQLITE_FLOAT;
         else if (declType == "TEXT") type = SQLITE_TEXT;
         else if (declType == "BLOB") type = SQLITE_BLOB;
         else throw std::runtime_error("Unexpected column decl type");
      }

      switch (type) {
      case SQLITE_INTEGER:
         fColumnTypes.push_back(Types::kInteger);
         fValues[i].fType = Types::kInteger;
         fValues[i].fPtr = &fValues[i].fInteger;
         break;
      case SQLITE_FLOAT:
         fColumnTypes.push_back(Types::kReal);
         fValues[i].fType = Types::kReal;
         fValues[i].fPtr = &fValues[i].fReal;
         break;
      case SQLITE_TEXT:
         fColumnTypes.push_back(Types::kText);
         fValues[i].fType = Types::kText;
         fValues[i].fPtr = &fValues[i].fText;
         break;
      case SQLITE_BLOB:
         fColumnTypes.push_back(Types::kBlob);
         fValues[i].fType = Types::kBlob;
         fValues[i].fPtr = &fValues[i].fBlob;
         break;
      case SQLITE_NULL:
         // TODO: Null values in first rows are not well handled
         fColumnTypes.push_back(Types::kNull);
         fValues[i].fType = Types::kNull;
         fValues[i].fPtr = &fValues[i].fNull;
         break;
      default:
         throw std::runtime_error("Unhandled data type");
      }
   }

   fTypeNames[Types::kInteger] = "Long64_t";
   fTypeNames[Types::kReal] = "double";
   fTypeNames[Types::kText] = "std::string";
   fTypeNames[Types::kBlob] = "std::vector<unsigned char>";
   fTypeNames[Types::kNull] = "void *";
}


RSqliteDS::~RSqliteDS()
{
   sqlite3_finalize(fQuery);
   sqlite3_close_v2(fDb);
}


const std::vector<std::string> &RSqliteDS::GetColumnNames() const
{
   return fColumnNames;
}


RDataSource::Record_t RSqliteDS::GetColumnReadersImpl(std::string_view name, const std::type_info &ti)
{
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   Types type = fColumnTypes[index];

   if ((type == Types::kReal && typeid(double) != ti) ||
       (type == Types::kInteger && typeid(Long64_t) != ti) ||
       (type == Types::kText && typeid(std::string) != ti) ||
       (type == Types::kBlob && typeid(std::vector<unsigned char>) != ti) ||
       (type == Types::kNull && typeid(void *) != ti))
   {
      std::string errmsg = "The type selected for column \"";
      errmsg += name;
      errmsg += "\" does not correspond to column type, which is ";
      errmsg += GetTypeName(name);
      throw std::runtime_error(errmsg);
   }

   fValues[index].fIsActive = true;
   std::vector<void *> result;
   for (unsigned i = 0; i < fNSlots; ++i) {
      result.push_back(&fValues[index].fPtr);
   }
   return result;
}


std::vector<std::pair<ULong64_t, ULong64_t>> RSqliteDS::GetEntryRanges()
{
   std::lock_guard<std::mutex> lockGuard(fLock);

   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   int retval = sqlite3_step(fQuery);
   switch (retval) {
   case SQLITE_DONE:
      return entryRanges;
   case SQLITE_ROW:
      entryRanges.emplace_back(fNRow, fNRow + 1);
      fNRow++;
      return entryRanges;
   default:
      SqliteError(retval);
      // Never here
      abort();
   }
}


std::string RSqliteDS::GetTypeName(std::string_view colName) const
{
   unsigned N = fColumnNames.size();

   for (unsigned i = 0; i < N; ++i) {
      if (colName == fColumnNames[i]) {
         return fTypeNames.at(fColumnTypes[i]);
      }
   }
   throw std::runtime_error("Unknown column: " + std::string(colName));
}


bool RSqliteDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) !=
          fColumnNames.end();
}


void RSqliteDS::Initialise() {
   fNRow = 0;
   int retval = sqlite3_reset(fQuery);
   if (retval != SQLITE_OK) throw std::runtime_error("SQlite error, reset");
}


RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query)
{
   ROOT::RDataFrame rdf(std::make_unique<RSqliteDS>(fileName, query));
   return rdf;
}


bool RSqliteDS::SetEntry(unsigned int /* slot */, ULong64_t entry)
{
   std::lock_guard<std::mutex> lockGuard(fLock);

   R__ASSERT(entry + 1 == fNRow);
   unsigned N = fValues.size();
   for (unsigned i = 0; i < N; ++i) {
     if (!fValues[i].fIsActive) continue;

     int nbytes;
     switch (fValues[i].fType) {
     case Types::kInteger:
        fValues[i].fInteger = sqlite3_column_int64(fQuery, i);
        break;
     case Types::kReal:
        fValues[i].fReal = sqlite3_column_double(fQuery, i);
        break;
     case Types::kText:
        nbytes = sqlite3_column_bytes(fQuery, i);
        if (nbytes == 0) {
           fValues[i].fText = "";
        } else {
           fValues[i].fText = reinterpret_cast<const char *>(
             sqlite3_column_text(fQuery, i));
        }
        break;
     case Types::kBlob:
        nbytes = sqlite3_column_bytes(fQuery, i);
        fValues[i].fBlob.resize(nbytes);
        if (nbytes > 0) {
           std::memcpy(fValues[i].fBlob.data(), sqlite3_column_blob(fQuery, i), nbytes);
        }
        break;
     case Types::kNull:
        break;
     default:
       throw std::runtime_error("Unhandled column type");
     }
   }
   return true;
}


void RSqliteDS::SetNSlots(unsigned int nSlots)
{
   fNSlots = nSlots;
}


void RSqliteDS::SqliteError(int errcode) {
   std::string errmsg = "SQlite error: ";
   errmsg += sqlite3_errstr(errcode);
   throw std::runtime_error(errmsg);
}

} // ns RDF

} // ns ROOT
