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

namespace ROOT {

namespace RDF {

RSqliteDS::RSqliteDS(std::string_view fileName, std::string_view query)
   : fDb(NULL)
   , fQuery(NULL)
   , fNRow(0)
{
   int retval;

   retval = sqlite3_open_v2(std::string(fileName).c_str(), &fDb,
     SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX, NULL);
   if (retval != SQLITE_OK) SqliteError(retval);

   retval = sqlite3_prepare_v2(fDb, std::string(query).c_str(), -1, &fQuery, NULL);
   if (retval != SQLITE_OK) SqliteError(retval);

   int colCount = sqlite3_column_count(fQuery);
   retval = sqlite3_step(fQuery);
   if ((retval != SQLITE_ROW) && (retval != SQLITE_DONE)) SqliteError(retval);

   for (int i = 0; i < colCount; ++i) {
      fValues.emplace_back(Value_t());
      fColumnNames.push_back(sqlite3_column_name(fQuery, i));
      int type = sqlite3_column_type(fQuery, i);
      if (retval == SQLITE_DONE) {
         fColumnTypes.push_back(Types::kNull);
         continue;
      }

      switch (type) {
         case SQLITE_INTEGER:
            fColumnTypes.push_back(Types::kInt);
            fValues[i].fType = Types::kInt;
            fValues[i].fPtr = &fValues[i].fInt;
            break;
         case SQLITE_FLOAT:
            fColumnTypes.push_back(Types::kFloat);
            fValues[i].fType = Types::kFloat;
            fValues[i].fPtr = &fValues[i].fFloat;
            break;
         case SQLITE_TEXT:
            fColumnTypes.push_back(Types::kText);
            fValues[i].fType = Types::kText;
            fValues[i].fPtr = &fValues[i].fText;
            break;
         case SQLITE_BLOB:
            fColumnTypes.push_back(Types::kBlob);
            break;
         case SQLITE_NULL:
            // TODO: Null values in first rows are not handled
            fColumnTypes.push_back(Types::kNull);
            break;
         default:
            throw std::runtime_error("Unhandled data type");
      }
   }
   retval = sqlite3_reset(fQuery);
   if (retval != SQLITE_OK) throw std::runtime_error("SQlite error");

   fTypeNames[Types::kInt] = "Long64_t";
   fTypeNames[Types::kFloat] = "double";
   fTypeNames[Types::kText] = "std::string";
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
   std::cout << "GetColumnReadersImpl " << name << std::endl;
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   std::cout << "  INDEX " << index << std::endl;
   Types type = fColumnTypes[index];

   if ((type == Types::kFloat && typeid(double) != ti) ||
       (type == Types::kInt && typeid(Long64_t) != ti) ||
       (type == Types::kText && typeid(std::string) != ti))
   {
      std::string errmsg = "The type selected for column \"";
      errmsg += name;
      errmsg += "\" does not correspond to column type, which is ";
      errmsg += GetTypeName(name);
      throw std::runtime_error(errmsg);
   }

   fValues[index].fIsActive = true;
   std::vector<void *> result;
   result.push_back(&fValues[index].fPtr);
   return result;
}


std::vector<std::pair<ULong64_t, ULong64_t>> RSqliteDS::GetEntryRanges()
{
   std::cout << "GetEntryRanges " << fNRow << std::endl;
   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   int retval = sqlite3_step(fQuery);
   switch (retval) {
      case SQLITE_DONE:
         return entryRanges;
      case SQLITE_ROW:
         entryRanges.emplace_back(fNRow, fNRow + 1);
         fNRow++;
         return entryRanges;
         break;
      default:
         SqliteError(retval);
         // Never here
         abort();
   }
}


std::string RSqliteDS::GetTypeName(std::string_view colName) const
{
   std::cout << "GetTypeName" << std::endl;
   unsigned N = fColumnNames.size();

   for (unsigned i = 0; i < N; ++i) {
      if (colName == fColumnNames[i]) {
         std::cout << "FOUND TYPE " << fTypeNames.at(fColumnTypes[i]) << std::endl;
         return fTypeNames.at(fColumnTypes[i]);
      }
   }
   std::string errmsg = "GetTypeName: " + std::string(colName) + " not found";
   throw std::runtime_error(errmsg);
}


bool RSqliteDS::HasColumn(std::string_view colName) const
{
   std::cout << "HasColumn" << std::endl;
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) !=
          fColumnNames.end();
}


RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query)
{
   ROOT::RDataFrame tdf(std::make_unique<RSqliteDS>(fileName, query));
   return tdf;
}


bool RSqliteDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   std::cout << "SetEntry " << entry << std::endl;
   if (slot != 0) throw std::runtime_error("unexpected slot id");
   R__ASSERT(entry + 1 == fNRow);
   unsigned N = fValues.size();
   std::cout << "N IS " << N << std::endl;
   for (unsigned i = 0; i < N; ++i) {
     if (!fValues[i].fIsActive) continue;
     std::cout << "OK for index " << i << std::endl;
     switch (fValues[i].fType) {
        case Types::kInt:
           fValues[i].fInt = sqlite3_column_int64(fQuery, i);
           std::cout << "setting " << i << " to " << fValues[i].fInt << std::endl;
           break;
        case Types::kFloat:
           fValues[i].fFloat = sqlite3_column_double(fQuery, i);
           break;
        case Types::kText:
           fValues[i].fText = reinterpret_cast<const char *>(
              sqlite3_column_text(fQuery, i));
           break;
        default:
          throw std::runtime_error("Unhandled column type");
     }
   }
   return true;
}


void RSqliteDS::SetNSlots(unsigned int nSlots)
{
   std::cout << "SetNSlots " << nSlots << std::endl;
   if (nSlots != 1) throw std::runtime_error("unexpected number of slots");
}


void RSqliteDS::SqliteError(int errcode) {
   std::string errmsg = "SQlite error: ";
   errmsg += sqlite3_errstr(errcode);
   throw std::runtime_error(errmsg);
}

} // ns RDF

} // ns ROOT
