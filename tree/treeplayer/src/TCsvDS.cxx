// Author: Enric Tejedor CERN  10/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::Experimental::TDF::TCsvDS
    \ingroup dataframe
    \brief TDataFrame data source class for reading CSV files.

The TCsvDS class implements a CSV file reader for TDataFrame.

A TDataFrame that reads from a CSV file can be constructed using the factory method
ROOT::Experimental::TDF::MakeCsvDataFrame, which accepts three parameters:
1. Path to the CSV file.
2. Boolean that specifies whether the first row of the CSV file contains headers or
not (optional, default `true`). If `false`, header names will be automatically generated.
3. Delimiter (optional, default ',').

The types of the columns in the CSV file are automatically inferred. The supported
types are:
- Integer: stored as a 64-bit long long int.
- Floating point number: stored with double precision.
- Boolean: matches the literals `true` and `false`.
- String: stored as an std::string, matches anything that does not fall into any of the
previous types.

These are some formatting rules expected by the TCsvDS implementation:
- All records must have the same number of fields, in the same order.
- Any field may be quoted.
~~~
    "1997","Ford","E350"
~~~
- Fields with embedded delimiters (e.g. comma) must be quoted.
~~~
    1997,Ford,E350,"Super, luxurious truck"
~~~
- Fields with double-quote characters must be quoted, and each of the embedded
double-quote characters must be represented by a pair of double-quote characters.
~~~
    1997,Ford,E350,"Super, ""luxurious"" truck"
~~~
- Fields with embedded line breaks are not supported, even when quoted.
~~~
    1997,Ford,E350,"Go get one now
    they are going fast"
~~~
- Spaces are considered part of a field and are not ignored.
~~~
    1997, Ford , E350
    not same as
    1997,Ford,E350
    but same as
    1997, "Ford" , E350
~~~
- If a header row is provided, it must contain column names for each of the fields.
~~~
    Year,Make,Model
    1997,Ford,E350
    2000,Mercury,Cougar
~~~

The current implementation of TCsvDS reads the entire CSV file content into memory before
TDataFrame starts processing it. Therefore, before creating a CSV TDataFrame, it is
important to check both how much memory is available and the size of the CSV file.
*/
// clang-format on

#include <ROOT/TDFUtils.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/TCsvDS.hxx>
#include <ROOT/memory.hxx>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

namespace ROOT {
namespace Experimental {
namespace TDF {

// Regular expressions for type inference
TRegexp TCsvDS::intRegex("^[-+]?[0-9]+$");
TRegexp TCsvDS::doubleRegex1("^[-+]?[0-9]+\\.[0-9]*$");
TRegexp TCsvDS::doubleRegex2("^[-+]?[0-9]*\\.[0-9]+$");
TRegexp TCsvDS::trueRegex("^true$");
TRegexp TCsvDS::falseRegex("^false$");

const std::map<TCsvDS::ColType_t, std::string>
   TCsvDS::fgColTypeMap({{'b', "bool"}, {'d', "double"}, {'l', "Long64_t"}, {'s', "std::string"}});

void TCsvDS::FillHeaders(const std::string &line)
{
   auto columns = ParseColumns(line);
   for (auto &col : columns) {
      fHeaders.emplace_back(col);
   }
}

void TCsvDS::FillRecord(const std::string &line, Record &record)
{
   std::istringstream lineStream(line);
   auto i = 0U;

   auto columns = ParseColumns(line);

   for (auto &col : columns) {
      auto colType = fColTypes[fHeaders[i]];

      switch (colType) {
      case 'd': {
         record.emplace_back(new double(std::stod(col)));
         break;
      }
      case 'l': {
         record.emplace_back(new Long64_t(std::stoll(col)));
         break;
      }
      case 'b': {
         auto b = new bool();
         record.emplace_back(b);
         std::istringstream is(col);
         is >> std::boolalpha >> *b;
         break;
      }
      case 's': {
         record.emplace_back(new std::string(col));
         break;
      }
      }
      ++i;
   }
}

void TCsvDS::GenerateHeaders(size_t size)
{
   for (size_t i = 0; i < size; ++i) {
      fHeaders.push_back("Col" + std::to_string(i));
   }
}

std::vector<void *> TCsvDS::GetColumnReadersImpl(std::string_view colName, const std::type_info &ti)
{
   const auto colType = GetType(colName);

   if ((colType == 'd' && typeid(double) != ti) || (colType == 'l' && typeid(Long64_t) != ti) ||
       (colType == 's' && typeid(std::string) != ti) || (colType == 'b' && typeid(bool) != ti)) {
      std::string err = "The type selected for column \"";
      err += colName;
      err += "\" does not correspond to column type, which is ";
      err += fgColTypeMap.at(colType);
      throw std::runtime_error(err);
   }

   const auto &colNames = GetColumnNames();
   const auto index = std::distance(colNames.begin(), std::find(colNames.begin(), colNames.end(), colName));
   std::vector<void *> ret(fNSlots);
   for (auto slot : ROOT::TSeqU(fNSlots)) {
      auto &val = fColAddresses[index][slot];
      if (ti == typeid(double)) {
         val = &fDoubleEvtValues[index][slot];
      } else if (ti == typeid(Long64_t)) {
         val = &fLong64EvtValues[index][slot];
      } else if (ti == typeid(std::string)) {
         val = &fStringEvtValues[index][slot];
      } else {
         val = &fBoolEvtValues[index][slot];
      }
      ret[slot] = &val;
   }
   return ret;
}

void TCsvDS::InferColTypes(std::vector<std::string> &columns)
{
   auto i = 0U;
   for (auto &col : columns) {
      InferType(col, i);
      ++i;
   }
}

void TCsvDS::InferType(const std::string &col, unsigned int idxCol)
{
   ColType_t type;
   int dummy;

   if (intRegex.Index(col, &dummy) != -1) {
      type = 'l'; // Long64_t
   } else if (doubleRegex1.Index(col, &dummy) != -1 || doubleRegex2.Index(col, &dummy) != -1) {
      type = 'd'; // double
   } else if (trueRegex.Index(col, &dummy) != -1 || falseRegex.Index(col, &dummy) != -1) {
      type = 'b'; // bool
   } else {       // everything else is a string
      type = 's'; // std::string
   }
   // TODO: Date

   fColTypes[fHeaders[idxCol]] = type;
   fColTypesList.push_back(type);
}

std::vector<std::string> TCsvDS::ParseColumns(const std::string &line)
{
   std::vector<std::string> columns;

   for (size_t i = 0; i < line.size(); ++i) {
      i = ParseValue(line, columns, i);
   }

   return columns;
}

size_t TCsvDS::ParseValue(const std::string &line, std::vector<std::string> &columns, size_t i)
{
   std::stringstream val;
   bool quoted = false;

   for (; i < line.size(); ++i) {
      if (line[i] == fDelimiter && !quoted) {
         break;
      } else if (line[i] == '"') {
         // Keep just one quote for escaped quotes, none for the normal quotes
         if (line[i + 1] != '"') {
            quoted = !quoted;
         } else {
            val << line[++i];
         }
      } else {
         val << line[i];
      }
   }

   columns.emplace_back(val.str());

   return i;
}

////////////////////////////////////////////////////////////////////////
/// Constructor to create a CSV TDataSource for TDataFrame.
/// \param[in] fileName Path of the CSV file.
/// \param[in] readHeaders `true` if the CSV file contains headers as first row, `false` otherwise
///                        (default `true`).
/// \param[in] delimiter Delimiter character (default ',').
TCsvDS::TCsvDS(std::string_view fileName, bool readHeaders, char delimiter) // TODO: Let users specify types?
   : fFileName(fileName),
     fDelimiter(delimiter)
{
   std::ifstream stream(fFileName);
   std::string line;

   // Read the headers if present
   if (readHeaders) {
      if (std::getline(stream, line)) {
         FillHeaders(line);
      } else {
         std::string msg = "Error reading headers of CSV file ";
         msg += fileName;
         throw std::runtime_error(msg);
      }
   }

   if (std::getline(stream, line)) {
      auto columns = ParseColumns(line);

      // Generate headers if not present
      if (!readHeaders) {
         GenerateHeaders(columns.size());
      }

      // Infer types of columns with first record
      InferColTypes(columns);

      // Read all records and store them in memory
      do {
         fRecords.emplace_back();
         FillRecord(line, fRecords.back());
      } while (std::getline(stream, line));
   }
}

////////////////////////////////////////////////////////////////////////
/// Destructor.
TCsvDS::~TCsvDS()
{
   for (auto &record : fRecords) {
      for (size_t i = 0; i < record.size(); ++i) {
         void *p = record[i];
         const auto colType = fColTypes[fHeaders[i]];
         switch (colType) {
         case 'd': {
            delete static_cast<double *>(p);
            break;
         }
         case 'l': {
            delete static_cast<Long64_t *>(p);
            break;
         }
         case 'b': {
            delete static_cast<bool *>(p);
            break;
         }
         case 's': {
            delete static_cast<std::string *>(p);
            break;
         }
         }
      }
   }
}

const std::vector<std::string> &TCsvDS::GetColumnNames() const
{
   return fHeaders;
}

std::vector<std::pair<ULong64_t, ULong64_t>> TCsvDS::GetEntryRanges()
{
   auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
   return entryRanges;
}

TCsvDS::ColType_t TCsvDS::GetType(std::string_view colName) const
{
   if (!HasColumn(colName)) {
      std::string msg = "The dataset does not have column ";
      msg += colName;
      throw std::runtime_error(msg);
   }

   return fColTypes.at(colName.data());
}

std::string TCsvDS::GetTypeName(std::string_view colName) const
{
   return fgColTypeMap.at(GetType(colName));
}

bool TCsvDS::HasColumn(std::string_view colName) const
{
   return fHeaders.end() != std::find(fHeaders.begin(), fHeaders.end(), colName);
}

void TCsvDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   int colIndex = 0;
   for (auto &colType : fColTypesList) {
      auto dataPtr = fRecords[entry][colIndex];
      switch (colType) {
      case 'd': {
         fDoubleEvtValues[colIndex][slot] = *static_cast<double *>(dataPtr);
         break;
      }
      case 'l': {
         fLong64EvtValues[colIndex][slot] = *static_cast<Long64_t *>(dataPtr);
         break;
      }
      case 'b': {
         fBoolEvtValues[colIndex][slot] = *static_cast<bool *>(dataPtr);
         break;
      }
      case 's': {
         fStringEvtValues[colIndex][slot] = *static_cast<std::string *>(dataPtr);
         break;
      }
      }
      colIndex++;
   }
}

void TCsvDS::SetNSlots(unsigned int nSlots)
{
   assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

   fNSlots = nSlots;

   const auto nColumns = fHeaders.size();
   // Initialise the entire set of addresses
   fColAddresses.resize(nColumns, std::vector<void *>(fNSlots, nullptr));

   // Initialize the per event data holders
   fDoubleEvtValues.resize(nColumns, std::vector<double>(fNSlots));
   fLong64EvtValues.resize(nColumns, std::vector<Long64_t>(fNSlots));
   fStringEvtValues.resize(nColumns, std::vector<std::string>(fNSlots));
   fBoolEvtValues.resize(nColumns, std::deque<bool>(fNSlots));
}

void TCsvDS::Initialise()
{
   const auto nRecords = fRecords.size();
   const auto chunkSize = nRecords / fNSlots;
   const auto remainder = 1U == fNSlots ? 0 : nRecords % fNSlots;
   auto start = 0UL;
   auto end = 0UL;

   for (auto i : ROOT::TSeqU(fNSlots)) {
      start = end;
      end += chunkSize;
      fEntryRanges.emplace_back(start, end);
      (void)i;
   }
   fEntryRanges.back().second += remainder;
}

TDataFrame MakeCsvDataFrame(std::string_view fileName, bool readHeaders, char delimiter)
{
   ROOT::Experimental::TDataFrame tdf(std::make_unique<TCsvDS>(fileName, readHeaders, delimiter));
   return tdf;
}

} // ns TDF
} // ns Experimental
} // ns ROOT
