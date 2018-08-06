// Author: Jakob Blomer CERN  07/2018

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSQLITEDS
#define ROOT_RSQLITEDS

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/RStringView.hxx"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <sqlite3.h>

namespace ROOT {

namespace RDF {

class RSqliteDS final : public ROOT::RDF::RDataSource {
private:
   enum class Types {
     kInteger,
     kReal,
     kText,
     kBlob,
     kNull
   };

   // Can be implemented by std::variant once available
   struct Value_t {
     explicit Value_t(Types type);

     Types fType;
     bool fIsActive;
     Long64_t fInteger;
     double fReal;
     std::string fText;
     std::vector<unsigned char> fBlob;
     void *fNull;
     void *fPtr;
   };

   void SqliteError(int errcode);

   sqlite3 *fDb;
   sqlite3_stmt *fQuery;
   unsigned int fNSlots;
   ULong64_t fNRow;
   std::vector<std::string> fColumnNames;
   std::vector<Types> fColumnTypes;
   std::map<Types, std::string> fTypeNames;
   std::vector<Value_t> fValues;
   std::mutex fLock;

public:
   RSqliteDS(std::string_view fileName, std::string_view query);
   ~RSqliteDS();
   void SetNSlots(unsigned int nSlots) final;
   const std::vector<std::string> &GetColumnNames() const final;
   bool HasColumn(std::string_view colName) const final;
   std::string GetTypeName(std::string_view colName) const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   bool SetEntry(unsigned int slot, ULong64_t entry) final;
   void Initialise() final;

protected:
   Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) final;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a SQlite RDataFrame.
/// \param[in] fileName Path of the sqlite file.
/// \param[in] query SQL query that defines the data set.
RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query);

} // ns RDF

} // ns ROOT

#endif
