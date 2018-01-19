// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h"      // R__USE_IMT
#include "ROOT/TDFNodes.hxx" // ColumnName2ColumnTypeName -> TCustomColumnBase, FindUnknownColumns -> TLoopManager
#include "ROOT/TDFUtils.hxx"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TClassRef.h"
#include "TFriendElement.h"
#include "TROOT.h" // IsImplicitMTEnabled, GetImplicitMTPoolSize

#include <stdexcept>
#include <string>
class TTree;
using namespace ROOT::Detail::TDF;
using namespace ROOT::Experimental::TDF;

namespace ROOT {
namespace Internal {
namespace TDF {

TIgnoreErrorLevelRAII::TIgnoreErrorLevelRAII(int errorIgnoreLevel)
{
   gErrorIgnoreLevel = errorIgnoreLevel;
}
TIgnoreErrorLevelRAII::~TIgnoreErrorLevelRAII()
{
   gErrorIgnoreLevel = fCurIgnoreErrorLevel;
}

/// Return a string containing the type of the given branch. Works both with real TTree branches and with temporary
/// column created by Define. Returns an empty string if type name deduction fails.
std::string
ColumnName2ColumnTypeName(const std::string &colName, TTree *tree, TCustomColumnBase *tmpBranch, TDataSource *ds)
{
   // if this is a TDataSource column, we just ask the type name to the data-source
   if (ds && ds->HasColumn(colName)) {
      return ds->GetTypeName(colName);
   }

   TBranch *branch = nullptr;
   if (tree)
      branch = tree->GetBranch(colName.c_str());
   if (branch) {
      // this must be a real TTree branch
      static const TClassRef tbranchelRef("TBranchElement");
      if (branch->InheritsFrom(tbranchelRef)) {
         // this branch is not a fundamental type, we can ask for the class name
         return static_cast<TBranchElement *>(branch)->GetClassName();
      } else {
         // this branch must be a fundamental type or array thereof
         const auto listOfLeaves = branch->GetListOfLeaves();
         const auto nLeaves = listOfLeaves->GetEntries();
         if (nLeaves != 1)
            throw std::runtime_error("TTree branch " + colName + " has " + std::to_string(nLeaves) +
                                     " leaves. Only one leaf per branch is supported.");
         TLeaf *l = static_cast<TLeaf *>(listOfLeaves->UncheckedAt(0));
         const std::string branchType = l->GetTypeName();
         if (branchType.empty()) {
            throw std::runtime_error("could not deduce type of branch " + std::string(colName));
         } else if (l->GetLeafCount() != nullptr && l->GetLenStatic() == 1) {
            // this is a variable-sized array
            return "ROOT::Experimental::TDF::TArrayBranch<" + branchType + ">";
         } else if (l->GetLeafCount() == nullptr && l->GetLenStatic() > 1) {
            // this is a fixed-sized array (we do not differentiate between variable- and fixed-sized arrays)
            return "ROOT::Experimental::TDF::TArrayBranch<" + branchType + ">";
         } else if (l->GetLeafCount() == nullptr && l->GetLenStatic() == 1) {
            // this branch contains a single fundamental type
            return l->GetTypeName();
         } else {
            // we do not know how to deal with this branch
            throw std::runtime_error("TTree branch " + colName + " has both a leaf count and a static length. This is not supported.");
         }
      }
   } else if (tmpBranch) {
      // this must be a temporary branch
      const auto &type_id = tmpBranch->GetTypeId();
      if (auto c = TClass::GetClass(type_id)) {
         return c->GetName();
      } else if (type_id == typeid(char))
         return "char";
      else if (type_id == typeid(unsigned char))
         return "unsigned char";
      else if (type_id == typeid(int))
         return "int";
      else if (type_id == typeid(unsigned int))
         return "unsigned int";
      else if (type_id == typeid(short))
         return "short";
      else if (type_id == typeid(unsigned short))
         return "unsigned short";
      else if (type_id == typeid(long))
         return "long";
      else if (type_id == typeid(unsigned long))
         return "unsigned long";
      else if (type_id == typeid(double))
         return "double";
      else if (type_id == typeid(float))
         return "float";
      else if (type_id == typeid(Long64_t))
         return "Long64_t";
      else if (type_id == typeid(ULong64_t))
         return "ULong64_t";
      else if (type_id == typeid(bool))
         return "bool";
      else {
         std::string msg("Cannot deduce type of temporary column ");
         msg += colName.c_str();
         msg += ". The typename is ";
         msg += tmpBranch->GetTypeId().name();
         msg += ".";
         throw std::runtime_error(msg);
      }
   } else {
      throw std::runtime_error("Column \"" + colName + "\" is not in a file and has not been defined.");
   }
}

const char *ToConstCharPtr(const char *s)
{
   return s;
}

const char *ToConstCharPtr(const std::string &s)
{
   return s.c_str();
}

unsigned int GetNSlots()
{
   unsigned int nSlots = 1;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      nSlots = ROOT::GetImplicitMTPoolSize();
#endif // R__USE_IMT
   return nSlots;
}

void GetBranchNamesImpl(TTree &t, std::set<std::string> &bNames, std::set<TTree *> &analysedTrees)
{

   if (!analysedTrees.insert(&t).second) {
      return;
   }

   auto branches = t.GetListOfBranches();
   if (branches) {
      for (auto branchObj : *branches) {
         bNames.insert(branchObj->GetName());
      }
   }

   auto friendTrees = t.GetListOfFriends();

   if (!friendTrees)
      return;

   for (auto friendTreeObj : *friendTrees) {
      auto friendTree = ((TFriendElement *)friendTreeObj)->GetTree();
      GetBranchNamesImpl(*friendTree, bNames, analysedTrees);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get all the branches names, including the ones of the friend trees
ColumnNames_t GetBranchNames(TTree &t)
{
   std::set<std::string> bNamesSet;
   std::set<TTree *> analysedTrees;
   GetBranchNamesImpl(t, bNamesSet, analysedTrees);
   ColumnNames_t bNames;
   for (auto &bName : bNamesSet)
      bNames.emplace_back(bName);
   return bNames;
}

void CheckCustomColumn(std::string_view definedCol, TTree *treePtr, const ColumnNames_t &customCols,
                       const ColumnNames_t &dataSourceColumns)
{
   const std::string definedColStr(definedCol);
   if (treePtr != nullptr) {
      // check if definedCol is already present in TTree
      const auto branch = treePtr->GetBranch(definedColStr.c_str());
      if (branch != nullptr) {
         const auto msg = "branch \"" + definedColStr + "\" already present in TTree";
         throw std::runtime_error(msg);
      }
   }
   // check if definedCol has already been `Define`d in the functional graph
   if (std::find(customCols.begin(), customCols.end(), definedCol) != customCols.end()) {
      const auto msg = "Redefinition of column \"" + definedColStr + "\"";
      throw std::runtime_error(msg);
   }
   // check if definedCol is already present in the DataSource (but has not yet been `Define`d)
   if (!dataSourceColumns.empty()) {
      if (std::find(dataSourceColumns.begin(), dataSourceColumns.end(), definedCol) != dataSourceColumns.end()) {
         const auto msg = "Redefinition of column \"" + definedColStr + "\" already present in the data-source";
         throw std::runtime_error(msg);
      }
   }
}

void CheckSnapshot(unsigned int nTemplateParams, unsigned int nColumnNames)
{
   if (nTemplateParams != nColumnNames) {
      std::string err_msg = "The number of template parameters specified for the snapshot is ";
      err_msg += std::to_string(nTemplateParams);
      err_msg += " while ";
      err_msg += std::to_string(nColumnNames);
      err_msg += " columns have been specified.";
      throw std::runtime_error(err_msg);
   }
}

/// Choose between local column names or default column names, throw in case of errors.
const ColumnNames_t
SelectColumns(unsigned int nRequiredNames, const ColumnNames_t &names, const ColumnNames_t &defaultNames)
{
   if (names.empty()) {
      // use default column names
      if (defaultNames.size() < nRequiredNames)
         throw std::runtime_error(
            std::to_string(nRequiredNames) + " column name" + (nRequiredNames == 1 ? " is" : "s are") +
            " required but none were provided and the default list has size " + std::to_string(defaultNames.size()));
      // return first nRequiredNames default column names
      return ColumnNames_t(defaultNames.begin(), defaultNames.begin() + nRequiredNames);
   } else {
      // use column names provided by the user to this particular transformation/action
      if (names.size() != nRequiredNames) {
         auto msg = std::to_string(nRequiredNames) + " column name" + (nRequiredNames == 1 ? " is" : "s are") +
                    " required but " + std::to_string(names.size()) + (names.size() == 1 ? " was" : " were") +
                    " provided:";
         for (const auto &name : names)
            msg += " \"" + name + "\",";
         msg.back() = '.';
         throw std::runtime_error(msg);
      }
      return names;
   }
}

ColumnNames_t FindUnknownColumns(const ColumnNames_t &requiredCols, TTree *tree, const ColumnNames_t &definedCols,
                                 const ColumnNames_t &dataSourceColumns)
{
   ColumnNames_t unknownColumns;
   for (auto &column : requiredCols) {
      const auto isTreeBranch = (tree != nullptr && tree->GetBranch(column.c_str()) != nullptr);
      if (isTreeBranch)
         continue;
      const auto isCustomColumn = std::find(definedCols.begin(), definedCols.end(), column) != definedCols.end();
      if (isCustomColumn)
         continue;
      const auto isDataSourceColumn =
         std::find(dataSourceColumns.begin(), dataSourceColumns.end(), column) != dataSourceColumns.end();
      if (isDataSourceColumn)
         continue;
      unknownColumns.emplace_back(column);
   }
   return unknownColumns;
}

bool IsInternalColumn(std::string_view colName)
{
   return 0 == colName.find("tdf") && '_' == colName.back();
}

} // end NS TDF
} // end NS Internal
} // end NS ROOT
