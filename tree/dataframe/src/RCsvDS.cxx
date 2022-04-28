// Author: Enric Tejedor CERN  10/2017

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::RDF::RCsvDS
    \ingroup dataframe
    \brief RDataFrame data source class for reading CSV files.

The RCsvDS class implements a CSV file reader for RDataFrame.

A RDataFrame that reads from a CSV file can be constructed using the factory method
ROOT::RDF::MakeCsvDataFrame, which accepts five parameters:
1. Path to the CSV file.
2. Boolean that specifies whether the first row of the CSV file contains headers or
not (optional, default `true`). If `false`, header names will be automatically generated as Col0, Col1, ..., ColN.
3. Delimiter (optional, default ',').
4. Chunk size (optional, default is -1 to read all) - number of lines to read at a time
5. Column Types (optional, default is an empty map). A map with column names as keys and their type
(expressed as a single character, see below) as values.

The type of columns that do not appear in the map is inferred from the data.
The supported types are:
- Integer: stored as a 64-bit long long int; can be specified in the column types map with 'L'.
- Floating point number: stored with double precision; specified with 'D'.
- Boolean: matches the literals `true` and `false`; specified with 'O'.
- String: stored as an std::string, matches anything that does not fall into any of the
previous types; specified with 'T'.

These are some formatting rules expected by the RCsvDS implementation:
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

The current implementation of RCsvDS reads the entire CSV file content into memory before
RDataFrame starts processing it. Therefore, before creating a CSV RDataFrame, it is
important to check both how much memory is available and the size of the CSV file.

RCsvDS can handle empty cells and also allows the usage of the special keywords "NaN" and "nan" to
indicate `nan` values. If the column is of type double, these cells are stored internally as `nan`.
Empty cells and explicit `nan`-s inside columns of type Long64_t/bool are stored as zeros.
*/
// clang-format on

#include <ROOT/RDF/Utils.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/RCsvDS.hxx>
#include <ROOT/RRawFile.hxx>
#include <TError.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace ROOT {

namespace RDF {

std::string RCsvDS::AsString()
{
   return "CSV data source";
}

// Regular expressions for type inference
const TRegexp RCsvDS::fgIntRegex("^[-+]?[0-9]+$");
const TRegexp RCsvDS::fgDoubleRegex1("^[-+]?[0-9]+\\.[0-9]*$");
const TRegexp RCsvDS::fgDoubleRegex2("^[-+]?[0-9]*\\.[0-9]+$");
const TRegexp RCsvDS::fgDoubleRegex3("^[-+]?[0-9]*\\.[0-9]+[eEdDqQ][-+]?[0-9]+$");
const TRegexp RCsvDS::fgTrueRegex("^true$");
const TRegexp RCsvDS::fgFalseRegex("^false$");

const std::unordered_map<RCsvDS::ColType_t, std::string>
   RCsvDS::fgColTypeMap({{'O', "bool"}, {'D', "double"}, {'L', "Long64_t"}, {'T', "std::string"}});

void RCsvDS::FillHeaders(const std::string &line)
{
   auto columns = ParseColumns(line);
   fHeaders.reserve(columns.size());
   for (auto &col : columns) {
      fHeaders.emplace_back(col);
   }
}

void RCsvDS::FillRecord(const std::string &line, Record_t &record)
{
   auto i = 0U;

   auto columns = ParseColumns(line);

   for (auto &col : columns) {
      auto colType = fColTypes[fHeaders[i]];

      switch (colType) {
      case 'D': {
         record.emplace_back(new double((col != "nan") ? std::stod(col) : std::numeric_limits<double>::quiet_NaN()));
         break;
      }
      case 'L': {
         if (col != "nan") {
            record.emplace_back(new Long64_t(std::stoll(col)));
         } else {
            fColContainingEmpty.insert(fHeaders[i]);
            record.emplace_back(new Long64_t(0));
         }
         break;
      }
      case 'O': {
         auto b = new bool();
         record.emplace_back(b);
         if (col != "nan") {
            std::istringstream(col) >> std::boolalpha >> *b;
         } else {
            fColContainingEmpty.insert(fHeaders[i]);
            *b = false;
         }
         break;
      }
      case 'T': {
         record.emplace_back(new std::string(col));
         break;
      }
      }
      ++i;
   }
}

void RCsvDS::GenerateHeaders(size_t size)
{
   fHeaders.reserve(size);
   for (size_t i = 0u; i < size; ++i) {
      fHeaders.push_back("Col" + std::to_string(i));
   }
}

std::vector<void *> RCsvDS::GetColumnReadersImpl(std::string_view colName, const std::type_info &ti)
{
   const auto colType = GetType(colName);

   if ((colType == 'D' && typeid(double) != ti) || (colType == 'L' && typeid(Long64_t) != ti) ||
       (colType == 'T' && typeid(std::string) != ti) || (colType == 'O' && typeid(bool) != ti)) {
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

void RCsvDS::ValidateColTypes(std::vector<std::string> &columns) const
{
   for (const auto &col : fColTypes) {
      if (!HasColumn(col.first)) {
         std::string msg = "There is no column with name \"" + col.first + "\".";
         if (!fReadHeaders) {
            msg += "\nSince the input csv file does not contain headers, valid column names";
            msg += " are [\"Col0\", ..., \"Col" + std::to_string(columns.size() - 1) + "\"].";
         }
         throw std::runtime_error(msg);
      }
      if (std::string("ODLT").find(col.second) == std::string::npos) {
         std::string msg = "Type alias '" + std::string(1, col.second) + "' is not supported.\n";
         msg += "Supported type aliases are 'O' for boolean, 'D' for double, 'L' for Long64_t, 'T' for std::string.";
         throw std::runtime_error(msg);
      }
   }
}

void RCsvDS::InferColTypes(std::vector<std::string> &columns)
{
   const auto second_line = fCsvFile->GetFilePos();

   for (auto i = 0u; i < columns.size(); ++i) {

      if (fColTypes.find(fHeaders[i]) != fColTypes.end())
         continue; // type was manually specified, nothing to do

      // read <=10 extra lines until a non-empty cell on this column is found, so that type is determined
      for (auto extraRowsRead = 0u; extraRowsRead < 10u && columns[i] == "nan"; ++extraRowsRead) {
         std::string line;
         if (!fCsvFile->Readln(line))
            break; // EOF
         const auto temp_columns = ParseColumns(line);
         if (temp_columns[i] != "nan")
            columns[i] = temp_columns[i]; // will break the loop in the next iteration
      }
      // reset the reading from the second line, because the first line is already loaded in `columns`
      fCsvFile->Seek(second_line);

      if (columns[i] == "nan") {
         // could not find a non-empty value, default to double
         fColTypes[fHeaders[i]] = 'D';
         fColTypesList.push_back('D');
      } else {
         InferType(columns[i], i);
      }
   }
}

void RCsvDS::InferType(const std::string &col, unsigned int idxCol)
{
   ColType_t type;
   int dummy;

   if (fgIntRegex.Index(col, &dummy) != -1) {
      type = 'L'; // Long64_t
   } else if (fgDoubleRegex1.Index(col, &dummy) != -1 || fgDoubleRegex2.Index(col, &dummy) != -1 ||
              fgDoubleRegex3.Index(col, &dummy) != -1) {
      type = 'D'; // double
   } else if (fgTrueRegex.Index(col, &dummy) != -1 || fgFalseRegex.Index(col, &dummy) != -1) {
      type = 'O'; // bool
   } else {       // everything else is a string
      type = 'T'; // std::string
   }
   // TODO: Date

   fColTypes[fHeaders[idxCol]] = type;
   fColTypesList.push_back(type);
}

std::vector<std::string> RCsvDS::ParseColumns(const std::string &line)
{
   std::vector<std::string> columns;

   for (size_t i = 0; i < line.size(); ++i) {
      i = ParseValue(line, columns, i);
   }

   return columns;
}

size_t RCsvDS::ParseValue(const std::string &line, std::vector<std::string> &columns, size_t i)
{
   std::string val;
   bool quoted = false;
   const size_t prevPos = i; // used to check if cell is empty

   for (; i < line.size(); ++i) {
      if (line[i] == fDelimiter && !quoted) {
         break;
      } else if (line[i] == '"') {
         // Keep just one quote for escaped quotes, none for the normal quotes
         if (line[i + 1] != '"') {
            quoted = !quoted;
         } else {
            val += line[++i];
         }
      } else {
         val += line[i];
      }
   }

   if (prevPos == i || val == "nan" || val == "NaN") // empty cell or explicit nan/NaN
      columns.emplace_back("nan");
   else
      columns.emplace_back(std::move(val));

   // if the line ends with the delimiter, we need to append the default column value
   // for the _next_, last column that won't be parsed (because we are out of characters)
   if (i == line.size() - 1 && line[i] == fDelimiter)
      columns.emplace_back("nan");

   return i;
}

////////////////////////////////////////////////////////////////////////
/// Constructor to create a CSV RDataSource for RDataFrame.
/// \param[in] fileName Path or URL of the CSV file.
/// \param[in] readHeaders `true` if the CSV file contains headers as first row, `false` otherwise
///                        (default `true`).
/// \param[in] delimiter Delimiter character (default ',').
/// \param[in] linesChunkSize bunch of lines to read, use -1 to read all
/// \param[in] colTypes Allows users to manually specify column types. Accepts an unordered map with keys being
///                     column names, values being type specifiers ('O' for boolean, 'D' for double, 'L' for
///                     Long64_t, 'T' for std::string)
RCsvDS::RCsvDS(std::string_view fileName, bool readHeaders, char delimiter, Long64_t linesChunkSize,
               std::unordered_map<std::string, char> &&colTypes)
   : fReadHeaders(readHeaders), fCsvFile(ROOT::Internal::RRawFile::Create(fileName)), fDelimiter(delimiter),
     fLinesChunkSize(linesChunkSize), fColTypes(std::move(colTypes))
{
   std::string line;

   // Read the headers if present
   if (fReadHeaders) {
      if (fCsvFile->Readln(line)) {
         FillHeaders(line);
      } else {
         std::string msg = "Error reading headers of CSV file ";
         msg += fileName;
         throw std::runtime_error(msg);
      }
   }

   fDataPos = fCsvFile->GetFilePos();
   bool eof = false;
   do {
      eof = !fCsvFile->Readln(line);
   } while (line.empty() && !eof);
   if (!eof) {
      auto columns = ParseColumns(line);

      // Generate headers if not present
      if (!fReadHeaders) {
         GenerateHeaders(columns.size());
      }

      // Ensure user is trying to set types only of existing columns
      ValidateColTypes(columns);

      // Infer types of columns with first record
      InferColTypes(columns);

      // rewind
      fCsvFile->Seek(fDataPos);
   } else {
      std::string msg = "Could not infer column types of CSV file ";
      msg += fileName;
      throw std::runtime_error(msg);
   }
}

void RCsvDS::FreeRecords()
{
   for (auto &record : fRecords) {
      for (size_t i = 0; i < record.size(); ++i) {
         void *p = record[i];
         const auto colType = fColTypes[fHeaders[i]];
         switch (colType) {
         case 'D': {
            delete static_cast<double *>(p);
            break;
         }
         case 'L': {
            delete static_cast<Long64_t *>(p);
            break;
         }
         case 'O': {
            delete static_cast<bool *>(p);
            break;
         }
         case 'T': {
            delete static_cast<std::string *>(p);
            break;
         }
         }
      }
   }
   fRecords.clear();
}

////////////////////////////////////////////////////////////////////////
/// Destructor.
RCsvDS::~RCsvDS()
{
   FreeRecords();
}

void RCsvDS::Finalize()
{
   fCsvFile->Seek(fDataPos);
   fProcessedLines = 0ULL;
   fEntryRangesRequested = 0ULL;
   FreeRecords();
}

const std::vector<std::string> &RCsvDS::GetColumnNames() const
{
   return fHeaders;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RCsvDS::GetEntryRanges()
{

   // Read records and store them in memory
   auto linesToRead = fLinesChunkSize;
   FreeRecords();

   std::string line;
   while ((-1LL == fLinesChunkSize || 0 != linesToRead) && fCsvFile->Readln(line)) {
      if (line.empty()) continue; // skip empty lines
      fRecords.emplace_back();
      FillRecord(line, fRecords.back());
      --linesToRead;
   }

   if (!fColContainingEmpty.empty()) {
      std::string msg = "";
      for (const auto &col : fColContainingEmpty) {
         const auto colT = GetTypeName(col);
         msg += "Column \"" + col + "\" of type " + colT + " contains empty cell(s) or NaN(s).\n";
         msg += "There is no `nan` equivalent for type " + colT + ", hence ";
         msg += std::string(colT == "Long64_t" ? "`0`" : "`false`") + " is stored.\n";
      }
      msg += "Please manually set the column type to `double` (with `D`) in `MakeCsvDataFrame` to read NaNs instead.\n";
      Warning("RCsvDS", "%s", msg.c_str());
   }

   if (gDebug > 0) {
      if (fLinesChunkSize == -1LL) {
         Info("GetEntryRanges", "Attempted to read entire CSV file into memory, %zu lines read", fRecords.size());
      } else {
         Info("GetEntryRanges", "Attempted to read chunk of %lld lines of CSV file into memory, %zu lines read", fLinesChunkSize, fRecords.size());
      }
   }

   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   const auto nRecords = fRecords.size();
   if (0 == nRecords)
      return entryRanges;

   const auto chunkSize = nRecords / fNSlots;
   const auto remainder = 1U == fNSlots ? 0 : nRecords % fNSlots;
   auto start = 0ULL == fEntryRangesRequested ? 0ULL : fProcessedLines;
   auto end = start;

   for (auto i : ROOT::TSeqU(fNSlots)) {
      start = end;
      end += chunkSize;
      entryRanges.emplace_back(start, end);
      (void)i;
   }
   entryRanges.back().second += remainder;

   fProcessedLines += nRecords;
   fEntryRangesRequested++;

   return entryRanges;
}

RCsvDS::ColType_t RCsvDS::GetType(std::string_view colName) const
{
   if (!HasColumn(colName)) {
      std::string msg = "The dataset does not have column ";
      msg += colName;
      throw std::runtime_error(msg);
   }

   return fColTypes.at(colName.data());
}

std::string RCsvDS::GetTypeName(std::string_view colName) const
{
   return fgColTypeMap.at(GetType(colName));
}

bool RCsvDS::HasColumn(std::string_view colName) const
{
   return fHeaders.end() != std::find(fHeaders.begin(), fHeaders.end(), colName);
}

bool RCsvDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   // Here we need to normalise the entry to the number of lines we already processed.
   const auto offset = (fEntryRangesRequested - 1) * fLinesChunkSize;
   const auto recordPos = entry - offset;
   int colIndex = 0;
   for (auto &colType : fColTypesList) {
      auto dataPtr = fRecords[recordPos][colIndex];
      switch (colType) {
      case 'D': {
         fDoubleEvtValues[colIndex][slot] = *static_cast<double *>(dataPtr);
         break;
      }
      case 'L': {
         fLong64EvtValues[colIndex][slot] = *static_cast<Long64_t *>(dataPtr);
         break;
      }
      case 'O': {
         fBoolEvtValues[colIndex][slot] = *static_cast<bool *>(dataPtr);
         break;
      }
      case 'T': {
         fStringEvtValues[colIndex][slot] = *static_cast<std::string *>(dataPtr);
         break;
      }
      }
      colIndex++;
   }
   return true;
}

void RCsvDS::SetNSlots(unsigned int nSlots)
{
   assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

   fNSlots = nSlots;

   const auto nColumns = fHeaders.size();
   // Initialize the entire set of addresses
   fColAddresses.resize(nColumns, std::vector<void *>(fNSlots, nullptr));

   // Initialize the per event data holders
   fDoubleEvtValues.resize(nColumns, std::vector<double>(fNSlots));
   fLong64EvtValues.resize(nColumns, std::vector<Long64_t>(fNSlots));
   fStringEvtValues.resize(nColumns, std::vector<std::string>(fNSlots));
   fBoolEvtValues.resize(nColumns, std::deque<bool>(fNSlots));
}

std::string RCsvDS::GetLabel()
{
   return "RCsv";
}

RDataFrame MakeCsvDataFrame(std::string_view fileName, bool readHeaders, char delimiter, Long64_t linesChunkSize,
                            std::unordered_map<std::string, char> &&colTypes)
{
   ROOT::RDataFrame rdf(
      std::make_unique<RCsvDS>(fileName, readHeaders, delimiter, linesChunkSize, std::move(colTypes)));
   return rdf;
}

} // ns RDF

} // ns ROOT
