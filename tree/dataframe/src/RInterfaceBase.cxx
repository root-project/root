// Author: Enrico Guiraud CERN 08/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/InternalTreeUtils.hxx> // GetFriendInfo, GetFileNamesFromTree
#include <ROOT/RDF/RDFDescription.hxx>
#include <ROOT/RDF/RInterfaceBase.hxx>
#include <ROOT/RDF/Utils.hxx>
#include <ROOT/RDF/RVariationsDescription.hxx>
#include <ROOT/RStringView.hxx>
#include <TTree.h>

#include <algorithm> // std::for_each
#include <iomanip>   // std::setw
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>

std::string ROOT::RDF::RInterfaceBase::DescribeDataset() const
{
   // TTree/TChain as input
   const auto tree = fLoopManager->GetTree();
   if (tree) {
      const auto treeName = tree->GetName();
      const auto isTChain = dynamic_cast<TChain *>(tree) ? true : false;
      const auto treeType = isTChain ? "TChain" : "TTree";
      const auto isInMemory = !isTChain && !tree->GetCurrentFile() ? true : false;
      const auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(*tree);
      const auto hasFriends = friendInfo.fFriendNames.empty() ? false : true;
      std::stringstream ss;
      ss << "Dataframe from " << treeType << " " << treeName;
      if (isInMemory) {
         ss << " (in-memory)";
      } else {
         const auto files = ROOT::Internal::TreeUtils::GetFileNamesFromTree(*tree);
         const auto numFiles = files.size();
         if (numFiles == 1) {
            ss << " in file " << files[0];
         } else {
            ss << " in files\n";
            for (auto i = 0u; i < numFiles; i++) {
               ss << "  " << files[i];
               if (i < numFiles - 1)
                  ss << '\n';
            }
         }
      }
      if (hasFriends) {
         const auto numFriends = friendInfo.fFriendNames.size();
         if (numFriends == 1) {
            ss << "\nwith friend\n";
         } else {
            ss << "\nwith friends\n";
         }
         for (auto i = 0u; i < numFriends; i++) {
            const auto nameAlias = friendInfo.fFriendNames[i];
            const auto files = friendInfo.fFriendFileNames[i];
            const auto numFiles = files.size();
            const auto subnames = friendInfo.fFriendChainSubNames[i];
            ss << "  " << nameAlias.first;
            if (nameAlias.first != nameAlias.second)
               ss << " (" << nameAlias.second << ")";
            // case: TTree as friend
            if (numFiles == 1) {
               ss << " " << files[0];
            }
            // case: TChain as friend
            else {
               ss << '\n';
               for (auto j = 0u; j < numFiles; j++) {
                  ss << "    " << subnames[j] << " " << files[j];
                  if (j < numFiles - 1)
                     ss << '\n';
               }
            }
            if (i < numFriends - 1)
               ss << '\n';
         }
      }
      return ss.str();
   }
   // Datasource as input
   else if (fDataSource) {
      const auto datasourceLabel = fDataSource->GetLabel();
      return "Dataframe from datasource " + datasourceLabel;
   }
   // Trivial/empty datasource
   else {
      const auto n = fLoopManager->GetNEmptyEntries();
      if (n == 1) {
         return "Empty dataframe filling 1 row";
      } else {
         return "Empty dataframe filling " + std::to_string(n) + " rows";
      }
   }
}

ROOT::RDF::RInterfaceBase::RInterfaceBase(std::shared_ptr<RDFDetail::RLoopManager> lm)
   : fLoopManager(lm.get()), fDataSource(lm->GetDataSource()), fColRegister(std::move(lm))
{
   AddDefaultColumns();
}

ROOT::RDF::RInterfaceBase::RInterfaceBase(RDFDetail::RLoopManager &lm, const RDFInternal::RColumnRegister &colRegister)
   : fLoopManager(&lm), fDataSource(lm.GetDataSource()), fColRegister(colRegister)
{
}

/////////////////////////////////////////////////////////////////////////////
/// \brief Returns the names of the available columns.
/// \return the container of column names.
///
/// This is not an action nor a transformation, just a query to the RDataFrame object.
///
/// ### Example usage:
/// ~~~{.cpp}
/// auto colNames = d.GetColumnNames();
/// // Print columns' names
/// for (auto &&colName : colNames) std::cout << colName << std::endl;
/// ~~~
///
ROOT::RDF::ColumnNames_t ROOT::RDF::RInterfaceBase::GetColumnNames()
{
   // there could be duplicates between Redefined columns and columns in the data source
   std::unordered_set<std::string> allColumns;

   auto addIfNotInternal = [&allColumns](std::string_view colName) {
      if (!RDFInternal::IsInternalColumn(colName))
         allColumns.emplace(colName);
   };

   auto definedColumns = fColRegister.GetNames();

   std::for_each(definedColumns.begin(), definedColumns.end(), addIfNotInternal);

   auto tree = fLoopManager->GetTree();
   if (tree) {
      for (const auto &bName : RDFInternal::GetBranchNames(*tree, /*allowDuplicates=*/false))
         allColumns.emplace(bName);
   }

   if (fDataSource) {
      for (const auto &s : fDataSource->GetColumnNames()) {
         if (s.rfind("R_rdf_sizeof", 0) != 0)
            allColumns.emplace(s);
      }
   }

   ColumnNames_t ret(allColumns.begin(), allColumns.end());
   std::sort(ret.begin(), ret.end());
   return ret;
}

/////////////////////////////////////////////////////////////////////////////
/// \brief Return the type of a given column as a string.
/// \return the type of the required column.
///
/// This is not an action nor a transformation, just a query to the RDataFrame object.
///
/// ### Example usage:
/// ~~~{.cpp}
/// auto colType = d.GetColumnType("columnName");
/// // Print column type
/// std::cout << "Column " << colType << " has type " << colType << std::endl;
/// ~~~
///
std::string ROOT::RDF::RInterfaceBase::GetColumnType(std::string_view column)
{
   const auto col = fColRegister.ResolveAlias(std::string(column));

   RDFDetail::RDefineBase *define = fColRegister.GetDefine(col);

   const bool convertVector2RVec = true;
   return RDFInternal::ColumnName2ColumnTypeName(col, fLoopManager->GetTree(), fLoopManager->GetDataSource(), define,
                                                 convertVector2RVec);
}

/////////////////////////////////////////////////////////////////////////////
/// \brief Return information about the dataframe.
/// \return information about the dataframe as RDFDescription object
///
/// This convenience function describes the dataframe and combines the following information:
/// - Number of event loops run, see GetNRuns()
/// - Number of total and defined columns, see GetColumnNames() and GetDefinedColumnNames()
/// - Column names, see GetColumnNames()
/// - Column types, see GetColumnType()
/// - Number of processing slots, see GetNSlots()
///
/// This is not an action nor a transformation, just a query to the RDataFrame object.
/// The result is dependent on the node from which this method is called, e.g. the list of
/// defined columns returned by GetDefinedColumnNames().
///
/// Please note that this is a convenience feature and the layout of the output can be subject
/// to change and should be parsed via RDFDescription methods.
///
/// ### Example usage:
/// ~~~{.cpp}
/// RDataFrame df(10);
/// auto df2 = df.Define("x", "1.f").Define("s", "\"myStr\"");
/// // Describe the dataframe
/// df2.Describe().Print()
/// df2.Describe().Print(/*shortFormat=*/true)
/// std::cout << df2.Describe().AsString() << std::endl;
/// std::cout << df2.Describe().AsString(/*shortFormat=*/true) << std::endl;
/// ~~~
///
ROOT::RDF::RDFDescription ROOT::RDF::RInterfaceBase::Describe()
{
   // Build set of defined column names to find later in all column names
   // the defined columns more efficiently
   const auto columnNames = GetColumnNames();
   std::set<std::string> definedColumnNamesSet;
   for (const auto &name : GetDefinedColumnNames())
      definedColumnNamesSet.insert(name);

   // Get information for the metadata table
   const std::vector<std::string> metadataProperties = {"Columns in total", "Columns from defines", "Event loops run",
                                                        "Processing slots"};
   const std::vector<std::string> metadataValues = {std::to_string(columnNames.size()),
                                                    std::to_string(definedColumnNamesSet.size()),
                                                    std::to_string(GetNRuns()), std::to_string(GetNSlots())};

   // Set header for metadata table
   const auto columnWidthProperties = RDFInternal::GetColumnWidth(metadataProperties);
   // The column width of the values is required to make right-bound numbers and is equal
   // to the maximum of the string "Value" and all values to be put in this column.
   const auto columnWidthValues =
      std::max(std::max_element(metadataValues.begin(), metadataValues.end())->size(), static_cast<std::size_t>(5u));
   std::stringstream ss;
   ss << std::left << std::setw(columnWidthProperties) << "Property" << std::setw(columnWidthValues) << "Value\n"
      << std::setw(columnWidthProperties) << "--------" << std::setw(columnWidthValues) << "-----\n";

   // Build metadata table
   // All numbers should be bound to the right and strings bound to the left.
   for (auto i = 0u; i < metadataProperties.size(); i++) {
      ss << std::left << std::setw(columnWidthProperties) << metadataProperties[i] << std::right
         << std::setw(columnWidthValues) << metadataValues[i] << '\n';
   }
   ss << '\n'; // put space between this and the next table

   // Set header for columns table
   const auto columnWidthNames = RDFInternal::GetColumnWidth(columnNames);
   const auto columnTypes = GetColumnTypeNamesList(columnNames);
   const auto columnWidthTypes = RDFInternal::GetColumnWidth(columnTypes);
   ss << std::left << std::setw(columnWidthNames) << "Column" << std::setw(columnWidthTypes) << "Type"
      << "Origin\n"
      << std::setw(columnWidthNames) << "------" << std::setw(columnWidthTypes) << "----"
      << "------\n";

   // Build columns table
   const auto nCols = columnNames.size();
   for (auto i = 0u; i < nCols; i++) {
      auto origin = "Dataset";
      if (definedColumnNamesSet.find(columnNames[i]) != definedColumnNamesSet.end())
         origin = "Define";
      ss << std::left << std::setw(columnWidthNames) << columnNames[i] << std::setw(columnWidthTypes) << columnTypes[i]
         << origin;
      if (i < nCols - 1)
         ss << '\n';
   }
   // Use the string returned from DescribeDataset() as the 'brief' description
   // Use the converted to string stringstream ss as the 'full' description
   return RDFDescription(DescribeDataset(), ss.str());
}

/// \brief Returns the names of the defined columns.
/// \return the container of the defined column names.
///
/// This is not an action nor a transformation, just a simple utility to
/// get the columns names that have been defined up to the node.
/// If no column has been defined, e.g. on a root node, it returns an
/// empty collection.
///
/// ### Example usage:
/// ~~~{.cpp}
/// auto defColNames = d.GetDefinedColumnNames();
/// // Print defined columns' names
/// for (auto &&defColName : defColNames) std::cout << defColName << std::endl;
/// ~~~
///
ROOT::RDF::ColumnNames_t ROOT::RDF::RInterfaceBase::GetDefinedColumnNames()
{
   ColumnNames_t definedColumns;

   const auto columns = fColRegister.BuildDefineNames();
   for (const auto &column : columns) {
      if (!RDFInternal::IsInternalColumn(column))
         definedColumns.emplace_back(column);
   }

   return definedColumns;
}

/// \brief Return a descriptor for the systematic variations registered in this branch of the computation graph.
///
/// This is not an action nor a transformation, just a simple utility to
/// inspect the systematic variations that have been registered with Vary() up to this node.
/// When called on the root node, it returns an empty descriptor.
///
/// ### Example usage:
/// ~~~{.cpp}
/// auto variations = d.GetVariations();
/// variations.Print();
/// ~~~
///
ROOT::RDF::RVariationsDescription ROOT::RDF::RInterfaceBase::GetVariations() const
{
   return fColRegister.BuildVariationsDescription();
}

/// \brief Checks if a column is present in the dataset.
/// \return true if the column is available, false otherwise
///
/// This method checks if a column is part of the input ROOT dataset, has
/// been defined or can be provided by the data source.
///
/// Example usage:
/// ~~~{.cpp}
/// ROOT::RDataFrame base(1);
/// auto rdf = base.Define("definedColumn", [](){return 0;});
/// rdf.HasColumn("definedColumn"); // true: we defined it
/// rdf.HasColumn("rdfentry_"); // true: it's always there
/// rdf.HasColumn("foo"); // false: it is not there
/// ~~~
bool ROOT::RDF::RInterfaceBase::HasColumn(std::string_view columnName)
{
   if (fColRegister.IsDefineOrAlias(columnName))
      return true;

   if (fLoopManager->GetTree()) {
      const auto &branchNames = fLoopManager->GetBranchNames();
      const auto branchNamesEnd = branchNames.end();
      if (branchNamesEnd != std::find(branchNames.begin(), branchNamesEnd, columnName))
         return true;
   }

   if (fDataSource && fDataSource->HasColumn(columnName))
      return true;

   return false;
}

/// \brief Gets the number of data processing slots.
/// \return The number of data processing slots used by this RDataFrame instance
///
/// This method returns the number of data processing slots used by this RDataFrame
/// instance. This number is influenced by the global switch ROOT::EnableImplicitMT().
///
/// Example usage:
/// ~~~{.cpp}
/// ROOT::EnableImplicitMT(6)
/// ROOT::RDataFrame df(1);
/// std::cout << df.GetNSlots() << std::endl; // prints "6"
/// ~~~
unsigned int ROOT::RDF::RInterfaceBase::GetNSlots() const
{
   return fLoopManager->GetNSlots();
}

/// \brief Gets the number of event loops run.
/// \return The number of event loops run by this RDataFrame instance
///
/// This method returns the number of events loops run so far by this RDataFrame instance.
///
/// Example usage:
/// ~~~{.cpp}
/// ROOT::RDataFrame df(1);
/// std::cout << df.GetNRuns() << std::endl; // prints "0"
/// df.Sum("rdfentry_").GetValue(); // trigger the event loop
/// std::cout << df.GetNRuns() << std::endl; // prints "1"
/// df.Sum("rdfentry_").GetValue(); // trigger another event loop
/// std::cout << df.GetNRuns() << std::endl; // prints "2"
/// ~~~
unsigned int ROOT::RDF::RInterfaceBase::GetNRuns() const
{
   return fLoopManager->GetNRuns();
}

ROOT::RDF::ColumnNames_t ROOT::RDF::RInterfaceBase::GetColumnTypeNamesList(const ColumnNames_t &columnList)
{
   std::vector<std::string> types;

   for (auto column : columnList) {
      types.push_back(GetColumnType(column));
   }
   return types;
}

void ROOT::RDF::RInterfaceBase::CheckIMTDisabled(std::string_view callerName)
{
   if (ROOT::IsImplicitMTEnabled()) {
      std::string error(callerName);
      error += " was called with ImplicitMT enabled, but multi-thread is not supported.";
      throw std::runtime_error(error);
   }
}

void ROOT::RDF::RInterfaceBase::AddDefaultColumns()
{
   // Entry number column
   const std::string entryColName = "rdfentry_";
   const std::string entryColType = "ULong64_t";
   auto entryColGen = [](unsigned int, ULong64_t entry) { return entry; };
   using NewColEntry_t = RDFDetail::RDefine<decltype(entryColGen), RDFDetail::ExtraArgsForDefine::SlotAndEntry>;

   auto entryColumn = std::make_shared<NewColEntry_t>(entryColName, entryColType, std::move(entryColGen),
                                                      ColumnNames_t{}, fColRegister, *fLoopManager);
   fColRegister.AddDefine(std::move(entryColumn));

   // Slot number column
   const std::string slotColName = "rdfslot_";
   const std::string slotColType = "unsigned int";
   auto slotColGen = [](unsigned int slot) { return slot; };
   using NewColSlot_t = RDFDetail::RDefine<decltype(slotColGen), RDFDetail::ExtraArgsForDefine::Slot>;

   auto slotColumn = std::make_shared<NewColSlot_t>(slotColName, slotColType, std::move(slotColGen), ColumnNames_t{},
                                                    fColRegister, *fLoopManager);
   fColRegister.AddDefine(std::move(slotColumn));

   fColRegister.AddAlias("tdfentry_", entryColName);
   fColRegister.AddAlias("tdfslot_", slotColName);
}
