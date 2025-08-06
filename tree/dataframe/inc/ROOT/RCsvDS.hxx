// Author: Enric Tejedor CERN  10/2017

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCSVDS
#define ROOT_RCSVDS

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"

#include <cstdint>
#include <deque>
#include <list>
#include <unordered_map>
#include <set>
#include <memory>
#include <vector>

#include <TRegexp.h>

namespace ROOT::Internal::RDF {
class R__CLING_PTRCHECK(off) RCsvDSColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   void *fValuePtr;
   void *GetImpl(Long64_t) final { return fValuePtr; }

public:
   RCsvDSColumnReader(void *valuePtr) : fValuePtr(valuePtr) {}
};
} // namespace ROOT::Internal::RDF

namespace ROOT {

namespace Internal {
class RRawFile;
}

namespace RDF {

class RCsvDS final : public ROOT::RDF::RDataSource {
public:
   /// Options that control how the CSV file is parsed
   struct ROptions {
      /// The first line describes the columns. The names are used as RDF column names
      /// unless fColumnNames is not empty, in which case it replaces the given names.
      /// If both, fHeaders is false and fColumnNames is empty, generic column names Col1.n.Col$n$ are used.
      bool fHeaders = true;
      char fDelimiter = ',';             ///< Column delimiter character
      bool fLeftTrim = false;            ///< Leading whitespaces are removed
      bool fRightTrim = false;           ///< Trailing whitespaces are removed
      bool fSkipBlankLines = true;       ///< Ignore empty lines (after trimming, if trimming is enabled)
      std::int64_t fSkipFirstNLines = 0; ///< Ignore the first N lines of the file
      std::int64_t fSkipLastNLines = 0;  ///< Ignore the last N lines of the file
      std::int64_t fLinesChunkSize = -1; ///< Number of lines to read, -1 to read all
      /// Character indicating that the remainder of the line should be ignored, if different from '\0'.
      /// If it is the first character of the line (after trimming), the line is ignored altogether.
      /// Note that the comment character must not be part of the data, e.g. in strings.
      char fComment = '\0';
      /// Impose column names. This can be used if a header is missing or if the header has unparsable or
      /// unwanted column names. If this list is not empty, it must contain exactly as many elements as
      /// the number of columns in the CSV file.
      std::vector<std::string> fColumnNames;
      /// Specify custom column types, accepts an unordered map with keys being column name, values being type alias
      /// ('O' for boolean, 'D' for double, 'L' for Long64_t, 'T' for std::string)
      std::unordered_map<std::string, char> fColumnTypes;
   };

private:
   // Possible values are D, O, L, T. This is possible only because we treat double, bool, Long64_t and string
   using ColType_t = char;
   static const std::unordered_map<ColType_t, std::string> fgColTypeMap;

   // Regular expressions for type inference
   static const TRegexp fgIntRegex, fgDoubleRegex1, fgDoubleRegex2, fgDoubleRegex3, fgTrueRegex, fgFalseRegex;

   ROptions fOptions;
   std::uint64_t fDataPos = 0;
   std::int64_t fDataLineNumber = 0;
   std::int64_t fLineNumber = 0;     // used to skip the last lines
   std::int64_t fMaxLineNumber = -1; // set to non-negative if fOptions.fSkipLastNLines is set
   std::unique_ptr<ROOT::Internal::RRawFile> fCsvFile;
   ULong64_t fEntryRangesRequested = 0ULL;
   ULong64_t fProcessedLines = 0ULL; // marks the progress of the consumption of the csv lines
   std::vector<std::string> fHeaders; // the column names
   std::unordered_map<std::string, ColType_t> fColTypes;
   std::set<std::string> fColContainingEmpty; // store columns which had empty entry
   std::list<ColType_t> fColTypesList; // column types, order is the same as fHeaders, values the same as fColTypes
   std::vector<std::vector<void *>> fColAddresses;         // fColAddresses[column][slot] (same ordering as fHeaders)
   std::vector<Record_t> fRecords;                         // fRecords[entry][column] (same ordering as fHeaders)
   std::vector<std::vector<double>> fDoubleEvtValues;      // one per column per slot
   std::vector<std::vector<Long64_t>> fLong64EvtValues;    // one per column per slot
   std::vector<std::vector<std::string>> fStringEvtValues; // one per column per slot
   // This must be a deque to avoid the specialisation vector<bool>. This would not
   // work given that the pointer to the boolean in that case cannot be taken
   std::vector<std::deque<bool>> fBoolEvtValues; // one per column per slot

   void Construct();

   bool Readln(std::string &line);
   void RewindToData();
   void FillHeaders(const std::string &);
   void FillRecord(const std::string &, Record_t &);
   void GenerateHeaders(size_t);
   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &) final;
   void ValidateColTypes(std::vector<std::string> &) const;
   void InferColTypes(std::vector<std::string> &);
   void InferType(const std::string &, unsigned int);
   std::vector<std::string> ParseColumns(const std::string &);
   size_t ParseValue(const std::string &, std::vector<std::string> &, size_t);
   ColType_t GetType(std::string_view colName) const;
   void FreeRecords();

protected:
   std::string AsString() final;

public:
   RCsvDS(std::string_view fileName, const ROptions &options);
   RCsvDS(std::string_view fileName, bool readHeaders = true, char delimiter = ',', Long64_t linesChunkSize = -1LL,
          std::unordered_map<std::string, char> &&colTypes = {});
   // Rule of five
   RCsvDS(const RCsvDS &) = delete;
   RCsvDS &operator=(const RCsvDS &) = delete;
   RCsvDS(RCsvDS &&) = delete;
   RCsvDS &operator=(RCsvDS &&) = delete;
   ~RCsvDS() final;

   void Finalize() final;
   std::size_t GetNFiles() const final { return 1; }
   const std::vector<std::string> &GetColumnNames() const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   std::string GetTypeName(std::string_view colName) const final;
   bool HasColumn(std::string_view colName) const final;
   bool SetEntry(unsigned int slot, ULong64_t entry) final;
   void SetNSlots(unsigned int nSlots) final;
   std::string GetLabel() final;

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int slot, std::string_view colName, const std::type_info &tid) final;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a CSV RDataFrame.
/// \param[in] fileName Path of the CSV file.
/// \param[in] options File parsing settings.
RDataFrame FromCSV(std::string_view fileName, const RCsvDS::ROptions &options);

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a CSV RDataFrame.
/// \param[in] fileName Path of the CSV file.
/// \param[in] readHeaders `true` if the CSV file contains headers as first row, `false` otherwise
///                        (default `true`).
/// \param[in] delimiter Delimiter character (default ',').
/// \param[in] linesChunkSize bunch of lines to read, use -1 to read all
/// \param[in] colTypes Allow user to specify custom column types, accepts an unordered map with keys being
///                      column type, values being type alias ('O' for boolean, 'D' for double, 'L' for
///                      Long64_t, 'T' for std::string)
RDataFrame FromCSV(std::string_view fileName, bool readHeaders = true, char delimiter = ',',
                   Long64_t linesChunkSize = -1LL, std::unordered_map<std::string, char> &&colTypes = {});

} // ns RDF

} // ns ROOT

#endif
