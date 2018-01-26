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

/// Return the type_info associated to a name. If the association fails, an
/// exception is thrown.
/// References and pointers are not supported since those cannot be stored in
/// columns.
const std::type_info &TypeName2TypeID(const std::string &name)
{
   if (auto c = TClass::GetClass(name.c_str())) {
      return *c->GetTypeInfo();
   } else if (name == "char" || name == "Char_t")
      return typeid(char);
   else if (name == "unsigned char" || name == "UChar_t")
      return typeid(unsigned char);
   else if (name == "int" || name == "Int_t")
      return typeid(int);
   else if (name == "unsigned int" || name == "UInt_t")
      return typeid(unsigned int);
   else if (name == "short" || name == "Short_t")
      return typeid(short);
   else if (name == "unsigned short" || name == "UShort_t")
      return typeid(unsigned short);
   else if (name == "long" || name == "Long_t")
      return typeid(long);
   else if (name == "unsigned long" || name == "ULong_t")
      return typeid(unsigned long);
   else if (name == "double" || name == "Double_t")
      return typeid(double);
   else if (name == "float" || name == "Float_t")
      return typeid(float);
   else if (name == "long long" || name == "long long int" || name == "Long64_t")
      return typeid(Long64_t);
   else if (name == "unsigned long long" || name == "unsigned long long int" || name == "ULong64_t")
      return typeid(ULong64_t);
   else if (name == "bool" || name == "Bool_t")
      return typeid(bool);
   else {
      std::string msg("Cannot extract type_info of type ");
      msg += name.c_str();
      msg += ".";
      throw std::runtime_error(msg);
   }
}

/// Returns the name of a type starting from its type_info
/// An empty string is returned in case of failure
/// References and pointers are not supported since those cannot be stored in
/// columns.
std::string TypeID2TypeName(const std::type_info &id)
{
   if (auto c = TClass::GetClass(id)) {
      return c->GetName();
   } else if (id == typeid(char))
      return "char";
   else if (id == typeid(unsigned char))
      return "unsigned char";
   else if (id == typeid(int))
      return "int";
   else if (id == typeid(unsigned int))
      return "unsigned int";
   else if (id == typeid(short))
      return "short";
   else if (id == typeid(unsigned short))
      return "unsigned short";
   else if (id == typeid(long))
      return "long";
   else if (id == typeid(unsigned long))
      return "unsigned long";
   else if (id == typeid(double))
      return "double";
   else if (id == typeid(float))
      return "float";
   else if (id == typeid(Long64_t))
      return "Long64_t";
   else if (id == typeid(ULong64_t))
      return "ULong64_t";
   else if (id == typeid(bool))
      return "bool";
   else
      return "";

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
            throw std::runtime_error("TTree branch " + colName +
                                     " has both a leaf count and a static length. This is not supported.");
         }
      }
   } else if (tmpBranch) {
      // this must be a temporary branch
      auto &id = tmpBranch->GetTypeId();
      auto typeName = TypeID2TypeName(id);
      if (typeName.empty()) {
         std::string msg("Cannot deduce type of temporary column ");
         msg += colName;
         msg += ". The typename is ";
         msg += id.name();
         msg += ".";
         throw std::runtime_error(msg);
      }
      return typeName;
   } else {
      throw std::runtime_error("Column \"" + colName + "\" is not in a file and has not been defined.");
   }
}

/// Convert type name (e.g. "Float_t") to ROOT type code (e.g. 'F') -- see TBranch documentation.
/// Return a space ' ' in case no match was found.
char TypeName2ROOTTypeName(const std::string &b)
{
   if (b == "Char_t")
      return 'B';
   if (b == "UChar_t")
      return 'b';
   if (b == "Short_t")
      return 'S';
   if (b == "UShort_t")
      return 's';
   if (b == "Int_t")
      return 'I';
   if (b == "UInt_t")
      return 'i';
   if (b == "Float_t")
      return 'F';
   if (b == "Double_t")
      return 'D';
   if (b == "Long64_t")
      return 'L';
   if (b == "ULong64_t")
      return 'l';
   if (b == "Bool_t")
      return 'O';
   return ' ';
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

// The set here is used as a registry, the real list, which keeps the order, is
// the one in the vector
void GetBranchNamesImpl(TTree &t, std::set<std::string> &bNamesReg, ColumnNames_t &bNames,
                        std::set<TTree *> &analysedTrees)
{

   if (!analysedTrees.insert(&t).second) {
      return;
   }

   auto branches = t.GetListOfBranches();
   if (branches) {
      for (auto branchObj : *branches) {
         auto name = branchObj->GetName();
         if (bNamesReg.insert(name).second) {
            bNames.emplace_back(name);
         }
      }
   }

   auto friendTrees = t.GetListOfFriends();

   if (!friendTrees)
      return;

   for (auto friendTreeObj : *friendTrees) {
      auto friendTree = ((TFriendElement *)friendTreeObj)->GetTree();
      GetBranchNamesImpl(*friendTree, bNamesReg, bNames, analysedTrees);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get all the branches names, including the ones of the friend trees
ColumnNames_t GetBranchNames(TTree &t)
{
   std::set<std::string> bNamesSet;
   ColumnNames_t bNames;
   std::set<TTree *> analysedTrees;
   GetBranchNamesImpl(t, bNamesSet, bNames, analysedTrees);
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
