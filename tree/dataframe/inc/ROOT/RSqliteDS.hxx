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

#include <map>
#include <memory>

#include <sqlite3.h>

namespace ROOT {

namespace RDF {

class RSqliteDS final : public ROOT::RDF::RDataSource {
private:
   enum class Types {
     kInt,
     kFloat,
     kText,
     kBlob,
     kNull
   };

   struct Value_t {
     Value_t() : fType(Types::kNull), fPtr(nullptr), fIsActive(false) {}

     Types fType;
     Long64_t fInt;
     double fFloat;
     std::string fText;
     void *fPtr;
     bool fIsActive;
   };

   void SqliteError(int errcode);

   struct sqlite3 *fDb;
   struct sqlite3_stmt *fQuery;
   ULong64_t fNRow;
   std::vector<std::string> fColumnNames;
   std::vector<Types> fColumnTypes;
   std::map<Types, std::string> fTypeNames;
   std::vector<Value_t> fValues;

public:
   RSqliteDS(std::string_view fileName, std::string_view query);
   ~RSqliteDS();
   virtual void SetNSlots(unsigned int nSlots) override;
   virtual const std::vector<std::string> &GetColumnNames() const override;
   virtual bool HasColumn(std::string_view colName) const override;
   virtual std::string GetTypeName(std::string_view colName) const override;
   virtual std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() override;
   virtual bool SetEntry(unsigned int slot, ULong64_t entry) override;
   virtual void Initialise() override;

protected:
   virtual Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) override;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a SQlite RDataFrame.
/// \param[in] fileName Path of the sqlite file.
/// \param[in] SQL query that defines the data set.
RDataFrame MakeSqliteDataFrame(std::string_view fileName, std::string_view query);

} // ns RDF

} // ns ROOT

#endif
