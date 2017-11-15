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
#include <ROOT/RMakeUnique.hxx>

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
      auto &colType = fColTypes[fHeaders[i]];

      if (colType == "Long64_t") {
         record.emplace_back(new Long64_t(std::stoll(col)));
      } else if (colType == "double") {
         record.emplace_back(new double(std::stod(col)));
      } else if (colType == "bool") {
         bool *b = new bool();
         record.emplace_back(b);
         std::istringstream is(col);
         is >> std::boolalpha >> *b;
      } else {
         record.emplace_back(new std::string(col));
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

std::vector<void *> TCsvDS::GetColumnReadersImpl(std::string_view colName, const std::type_info &)
{
   const auto &colNames = GetColumnNames();
   const auto index = std::distance(colNames.begin(), std::find(colNames.begin(), colNames.end(), colName));
   std::vector<void *> ret(fNSlots);
   for (auto slot : ROOT::TSeqU(fNSlots)) {
      ret[slot] = (void *)&fColAddresses[index][slot];
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
   std::string type;
   int dummy;

   if (intRegex.Index(col, &dummy) != -1) {
      type = "Long64_t";
   } else if (doubleRegex1.Index(col, &dummy) != -1 || doubleRegex2.Index(col, &dummy) != -1) {
      type = "double";
   } else if (trueRegex.Index(col, &dummy) != -1 || falseRegex.Index(col, &dummy) != -1) {
      type = "bool";
   } else { // everything else is a string
      type = "std::string";
   }
   // TODO: Date

   fColTypes[fHeaders[idxCol]] = type;
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
         auto &colType = fColTypes[fHeaders[i]];

         if (colType == "Long64_t") {
            delete static_cast<Long64_t *>(p);
         } else if (colType == "double") {
            delete static_cast<double *>(p);
         } else if (colType == "bool") {
            delete static_cast<bool *>(p);
         } else {
            delete static_cast<std::string *>(p);
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

std::string TCsvDS::GetTypeName(std::string_view colName) const
{
   if (!HasColumn(colName)) {
      std::string msg = "The dataset does not have column ";
      msg += colName;
      throw std::runtime_error(msg);
   }

   return fColTypes.at(colName.data());
}

bool TCsvDS::HasColumn(std::string_view colName) const
{
   return fHeaders.end() != std::find(fHeaders.begin(), fHeaders.end(), colName);
}

void TCsvDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   auto nColumns = fHeaders.size();

   for (auto i : ROOT::TSeqU(nColumns)) {
      // Update the address of every column of the slot to point to the record
      fColAddresses[i][slot] = fRecords[entry][i];
   }
}

void TCsvDS::SetNSlots(unsigned int nSlots)
{
   assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

   fNSlots = nSlots;

   const auto nColumns = fHeaders.size();
   // Initialise the entire set of addresses
   fColAddresses.resize(nColumns, std::vector<void *>(fNSlots, nullptr));
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
