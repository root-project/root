// Author: Enric Tejedor CERN  10/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCSVTDS
#define ROOT_TCSVTDS

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"

#include <deque>
#include <list>
#include <map>
#include <vector>

#include <TRegexp.h>

namespace ROOT {
namespace Experimental {
namespace TDF {

class TCsvDS final : public ROOT::Experimental::TDF::TDataSource {

private:
   using Record = std::vector<void *>;
   // Possible values are d, b, l, s. This is possible only because we treat double, bool, Long64_t and string
   using ColType_t = char;
   static const std::map<ColType_t, std::string> fgColTypeMap;

   unsigned int fNSlots = 0U;
   std::string fFileName;
   char fDelimiter;
   std::vector<std::string> fHeaders;
   std::map<std::string, ColType_t> fColTypes;
   std::list<ColType_t> fColTypesList;
   std::vector<std::vector<void *>> fColAddresses; // fColAddresses[column][slot]
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<Record> fRecords;                           // fRecords[entry][column]
   std::vector<std::vector<double>> fDoubleEvtValues;      // one per column per slot
   std::vector<std::vector<Long64_t>> fLong64EvtValues;    // one per column per slot
   std::vector<std::vector<std::string>> fStringEvtValues; // one per column per slot
   // This must be a deque to avoid the specialisation vector<bool>. This would not
   // work given that the pointer to the boolean in that case cannot be taken
   std::vector<std::deque<bool>> fBoolEvtValues; // one per column per slot

   static TRegexp intRegex, doubleRegex1, doubleRegex2, trueRegex, falseRegex;

   void FillHeaders(const std::string &);
   void FillRecord(const std::string &, Record &);
   void GenerateHeaders(size_t);
   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &);
   void InferColTypes(std::vector<std::string> &);
   void InferType(const std::string &, unsigned int);
   std::vector<std::string> ParseColumns(const std::string &);
   size_t ParseValue(const std::string &, std::vector<std::string> &, size_t);
   ColType_t GetType(std::string_view colName) const;

public:
   TCsvDS(std::string_view fileName, bool readHeaders = true, char delimiter = ',');
   ~TCsvDS();
   const std::vector<std::string> &GetColumnNames() const;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges();
   std::string GetTypeName(std::string_view colName) const;
   bool HasColumn(std::string_view colName) const;
   void SetEntry(unsigned int slot, ULong64_t entry);
   void SetNSlots(unsigned int nSlots);
   void Initialise();
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a CSV TDataFrame.
/// \param[in] fileName Path of the CSV file.
/// \param[in] readHeaders `true` if the CSV file contains headers as first row, `false` otherwise
///                        (default `true`).
/// \param[in] delimiter Delimiter character (default ',').
TDataFrame MakeCsvDataFrame(std::string_view fileName, bool readHeaders = true, char delimiter = ',');

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif
