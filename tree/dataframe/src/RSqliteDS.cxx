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
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace ROOT {

namespace RDF {

RSqliteDS::Value_t::Value_t(RSqliteDS::ETypes type)
   : fType(type)
   , fIsActive(false)
   , fInteger(0)
   , fReal(0.0)
   , fText()
   , fBlob()
   , fNull(nullptr)
{
   switch (type) {
   case ETypes::kInteger: fPtr = &fInteger; break;
   case ETypes::kReal: fPtr = &fReal; break;
   case ETypes::kText: fPtr = &fText; break;
   case ETypes::kBlob: fPtr = &fBlob; break;
   case ETypes::kNull: fPtr = &fNull; break;
   default: throw std::runtime_error("Internal error");
   }
}

constexpr char const *RSqliteDS::fgTypeNames[];

////////////////////////////////////////////////////////////////////////////
/// \brief Build the dataframe
/// \param[in] fileName The path to an sqlite3 file, will be opened read-only
/// \param[in] query A valid sqlite3 SELECT query
///
/// The constructor opens the sqlite file, prepares the query engine and determines the column names and types.
RSqliteDS::RSqliteDS(std::string_view fileName, std::string_view query)
   : fDb(nullptr), fQuery(nullptr), fNSlots(0), fNRow(0)
{
   int retval;

   retval = sqlite3_open_v2(std::string(fileName).c_str(), &fDb, SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, nullptr);
   if (retval != SQLITE_OK)
      SqliteError(retval);

   retval = sqlite3_prepare_v2(fDb, std::string(query).c_str(), -1, &fQuery, nullptr);
   if (retval != SQLITE_OK)
      SqliteError(retval);

   int colCount = sqlite3_column_count(fQuery);
   retval = sqlite3_step(fQuery);
   if ((retval != SQLITE_ROW) && (retval != SQLITE_DONE))
      SqliteError(retval);

   fValues.reserve(colCount);
   for (int i = 0; i < colCount; ++i) {
      fColumnNames.emplace_back(sqlite3_column_name(fQuery, i));
      int type = SQLITE_NULL;
      // Try first with the declared column type and then with the dynamic type
      // for expressions
      const char *declTypeCstr = sqlite3_column_decltype(fQuery, i);
      if (declTypeCstr == nullptr) {
         if (retval == SQLITE_ROW)
            type = sqlite3_column_type(fQuery, i);
      } else {
         std::string declType(declTypeCstr);
         std::transform(declType.begin(), declType.end(), declType.begin(), ::toupper);
         if (declType == "INTEGER")
            type = SQLITE_INTEGER;
         else if (declType == "FLOAT")
            type = SQLITE_FLOAT;
         else if (declType == "TEXT")
            type = SQLITE_TEXT;
         else if (declType == "BLOB")
            type = SQLITE_BLOB;
         else
            throw std::runtime_error("Unexpected column decl type");
      }

      switch (type) {
      case SQLITE_INTEGER:
         fColumnTypes.push_back(ETypes::kInteger);
         fValues.emplace_back(ETypes::kInteger);
         break;
      case SQLITE_FLOAT:
         fColumnTypes.push_back(ETypes::kReal);
         fValues.emplace_back(ETypes::kReal);
         break;
      case SQLITE_TEXT:
         fColumnTypes.push_back(ETypes::kText);
         fValues.emplace_back(ETypes::kText);
         break;
      case SQLITE_BLOB:
         fColumnTypes.push_back(ETypes::kBlob);
         fValues.emplace_back(ETypes::kBlob);
         break;
      case SQLITE_NULL:
         // TODO: Null values in first rows are not well handled
         fColumnTypes.push_back(ETypes::kNull);
         fValues.emplace_back(ETypes::kNull);
         break;
      default: throw std::runtime_error("Unhandled data type");
      }
   }
}

////////////////////////////////////////////////////////////////////////////
/// Frees the sqlite resources and closes the file.
RSqliteDS::~RSqliteDS()
{
   // sqlite3_finalize returns the error code of the most recent operation on fQuery.
   (void) sqlite3_finalize(fQuery);
   // Closing can possibly fail with SQLITE_BUSY, in which case resources are leaked. This should not happen
   // the way it is used in this class because we cleanup the prepared statement before.
   (void) sqlite3_close_v2(fDb);
}

////////////////////////////////////////////////////////////////////////////
/// Returns the SELECT queries names. The column names have been cached in the constructor.
/// For expressions, the column name is the string of the expression unless the query defines a column name with as
/// like in "SELECT 1 + 1 as mycolumn FROM table"
const std::vector<std::string> &RSqliteDS::GetColumnNames() const
{
   return fColumnNames;
}

////////////////////////////////////////////////////////////////////////////
/// Activates the given column's result value.
RDataSource::Record_t RSqliteDS::GetColumnReadersImpl(std::string_view name, const std::type_info &ti)
{
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   const auto type = fColumnTypes[index];

   if ((type == ETypes::kInteger && typeid(Long64_t) != ti) ||
       (type == ETypes::kReal && typeid(double) != ti) ||
       (type == ETypes::kText && typeid(std::string) != ti) ||
       (type == ETypes::kBlob && typeid(std::vector<unsigned char>) != ti) ||
       (type == ETypes::kNull && typeid(void *) != ti)) {
      std::string errmsg = "The type selected for column \"";
      errmsg += name;
      errmsg += "\" does not correspond to column type, which is ";
      errmsg += GetTypeName(name);
      throw std::runtime_error(errmsg);
   }

   fValues[index].fIsActive = true;
   return std::vector<void *>{fNSlots, &fValues[index].fPtr};
}

////////////////////////////////////////////////////////////////////////////
/// Returns a range of size 1 as long as more rows are available in the SQL result set.
/// This inherently serialized the RDF independent of the number of slots.
std::vector<std::pair<ULong64_t, ULong64_t>> RSqliteDS::GetEntryRanges()
{
   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   int retval = sqlite3_step(fQuery);
   switch (retval) {
   case SQLITE_DONE: return entryRanges;
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

////////////////////////////////////////////////////////////////////////////
/// Returns the C++ type for a given column name, implemented as a linear search through all the columns.
std::string RSqliteDS::GetTypeName(std::string_view colName) const
{
   unsigned N = fColumnNames.size();

   for (unsigned i = 0; i < N; ++i) {
      if (colName == fColumnNames[i]) {
         return fgTypeNames[static_cast<int>(fColumnTypes[i])];
      }
   }
   throw std::runtime_error("Unknown column: " + std::string(colName));
}

////////////////////////////////////////////////////////////////////////////
/// A linear search through the columns for the given name
bool RSqliteDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) != fColumnNames.end();
}

////////////////////////////////////////////////////////////////////////////
/// Resets the SQlite query engine at the beginning of the event loop.
void RSqliteDS::Initialise()
{
   fNRow = 0;
   int retval = sqlite3_reset(fQuery);
   if (retval != SQLITE_OK)
      throw std::runtime_error("SQlite error, reset");
}

std::string RSqliteDS::GetDataSourceType()
{
   return "RSqliteDS";
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a SQlite RDataFrame.
/// \param[in] fileName Path of the sqlite file.
/// \param[in] query SQL query that defines the data set.
RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query)
{
   ROOT::RDataFrame rdf(std::make_unique<RSqliteDS>(fileName, query));
   return rdf;
}

////////////////////////////////////////////////////////////////////////////
/// Stores the result of the current active sqlite query row as a C++ value.
bool RSqliteDS::SetEntry(unsigned int /* slot */, ULong64_t entry)
{
   R__ASSERT(entry + 1 == fNRow);
   unsigned N = fValues.size();
   for (unsigned i = 0; i < N; ++i) {
      if (!fValues[i].fIsActive)
         continue;

      int nbytes;
      switch (fValues[i].fType) {
      case ETypes::kInteger: fValues[i].fInteger = sqlite3_column_int64(fQuery, i); break;
      case ETypes::kReal: fValues[i].fReal = sqlite3_column_double(fQuery, i); break;
      case ETypes::kText:
         nbytes = sqlite3_column_bytes(fQuery, i);
         if (nbytes == 0) {
            fValues[i].fText = "";
         } else {
            fValues[i].fText = reinterpret_cast<const char *>(sqlite3_column_text(fQuery, i));
         }
         break;
      case ETypes::kBlob:
         nbytes = sqlite3_column_bytes(fQuery, i);
         fValues[i].fBlob.resize(nbytes);
         if (nbytes > 0) {
            std::memcpy(fValues[i].fBlob.data(), sqlite3_column_blob(fQuery, i), nbytes);
         }
         break;
      case ETypes::kNull: break;
      default: throw std::runtime_error("Unhandled column type");
      }
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Almost a no-op, many slots can in fact reduce the performance due to thread synchronization.
void RSqliteDS::SetNSlots(unsigned int nSlots)
{
   if (nSlots > 1) {
      ::Warning("SetNSlots", "Currently the SQlite data source faces performance degradation in multi-threaded mode. "
                             "Consider turning off IMT.");
   }
   fNSlots = nSlots;
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Helper function to throw an exception if there is a fatal sqlite error, e.g. an I/O error.
void RSqliteDS::SqliteError(int errcode)
{
   std::string errmsg = "SQlite error: ";
   errmsg += sqlite3_errstr(errcode);
   throw std::runtime_error(errmsg);
}

} // namespace RDF

} // namespace ROOT
