#ifndef ROOT_TCSVTDS
#define ROOT_TCSVTDS

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"

#include <map>
#include <vector>
#include <TRegexp.h>

namespace ROOT {
namespace Experimental {
namespace TDF {

class TCsvDS final : public ROOT::Experimental::TDF::TDataSource {

private:
   typedef std::vector<void *> Record;

   unsigned int fNSlots = 0U;
   std::string fFileName;
   char fDelimiter;
   std::vector<std::string> fHeaders;
   std::map<std::string, std::string> fColTypes;
   std::vector<std::vector<void *>> fColAddresses; // fColAddresses[column][slot]
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<Record> fRecords; // fRecords[entry][column]

   static TRegexp intRegex, doubleRegex1, doubleRegex2, trueRegex, falseRegex;
   
   void FillHeaders(const std::string &);
   void FillRecord(const std::string &, Record &);
   void GenerateHeaders(size_t);
   std::vector<void *> GetColumnReadersImpl(std::string_view, const std::type_info &);
   void InferColTypes(std::vector<std::string> &);
   void InferType(const std::string &, unsigned int);
   std::vector<std::string> ParseColumns(const std::string &);
   size_t ParseValue(const std::string &, std::vector<std::string> &, size_t);

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
