// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDFInterfaceUtils.hxx>
#include <ROOT/RStringView.hxx>
#include <RtypesCore.h>
#include <TClass.h>
#include <TFriendElement.h>
#include <TInterpreter.h>
#include <TObject.h>
#include <TRegexp.h>
#include <TString.h>
#include <TTree.h>
#include <TBranchElement.h>

#include <iosfwd>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace ROOT {
namespace Detail {
namespace RDF {
class RCustomColumnBase;
class RFilterBase;
class RLoopManager;
class RRangeBase;
} // namespace RDF
} // namespace Detail

namespace RDF {
class RDataSource;
} // namespace RDF

} // namespace ROOT

namespace ROOT {
namespace Internal {
namespace RDF {

// The set here is used as a registry, the real list, which keeps the order, is
// the one in the vector
class RActionBase;

bool ContainsLeaf(const std::set<TLeaf *> &leaves, TLeaf *leaf)
{
   return (leaves.find(leaf) != leaves.end());
}

///////////////////////////////////////////////////////////////////////////////
/// This overload does not perform any check on the duplicates.
/// It is used for TBranch objects.
void UpdateList(std::set<std::string> &bNamesReg, ColumnNames_t &bNames, std::string &branchName,
                std::string &friendName)
{

   if (!friendName.empty()) {
      // In case of a friend tree, users might prepend its name/alias to the branch names
      auto friendBName = friendName + "." + branchName;
      if (bNamesReg.insert(friendBName).second)
         bNames.push_back(friendBName);
   }

   if (bNamesReg.insert(branchName).second)
      bNames.push_back(branchName);
}

///////////////////////////////////////////////////////////////////////////////
/// This overloads makes sure that the TLeaf has not been already inserted.
void UpdateList(std::set<std::string> &bNamesReg, ColumnNames_t &bNames, std::string &branchName,
                std::string &friendName, std::set<TLeaf *> &foundLeaves, TLeaf *leaf, bool allowDuplicates)
{
   const bool canAdd = allowDuplicates ? true : !ContainsLeaf(foundLeaves, leaf);
   if (!canAdd) {
      return;
   }

   UpdateList(bNamesReg, bNames, branchName, friendName);

   foundLeaves.insert(leaf);
}

void ExploreBranch(TTree &t, std::set<std::string> &bNamesReg, ColumnNames_t &bNames, TBranch *b, std::string prefix,
                   std::string &friendName)
{
   for (auto sb : *b->GetListOfBranches()) {
      TBranch *subBranch = static_cast<TBranch *>(sb);
      auto subBranchName = std::string(subBranch->GetName());
      auto fullName = prefix + subBranchName;

      std::string newPrefix;
      if (!prefix.empty())
         newPrefix = fullName + ".";

      ExploreBranch(t, bNamesReg, bNames, subBranch, newPrefix, friendName);

      if (t.GetBranch(fullName.c_str())) {
         UpdateList(bNamesReg, bNames, fullName, friendName);

      } else if (t.GetBranch(subBranchName.c_str())) {
         UpdateList(bNamesReg, bNames, subBranchName, friendName);
      }
   }
}

void GetBranchNamesImpl(TTree &t, std::set<std::string> &bNamesReg, ColumnNames_t &bNames,
                        std::set<TTree *> &analysedTrees, std::string &friendName, bool allowDuplicates)
{
   std::set<TLeaf *> foundLeaves;
   if (!analysedTrees.insert(&t).second) {
      return;
   }

   const auto branches = t.GetListOfBranches();
   if (branches) {
      std::string prefix = "";
      for (auto b : *branches) {
         TBranch *branch = static_cast<TBranch *>(b);
         auto branchName = std::string(branch->GetName());
         if (branch->IsA() == TBranch::Class()) {
            // Leaf list
            auto listOfLeaves = branch->GetListOfLeaves();
            if (listOfLeaves->GetEntries() == 1) {
               auto leaf = static_cast<TLeaf *>(listOfLeaves->At(0));
               auto leafName = std::string(leaf->GetName());
               if (leafName == branchName) {
                  UpdateList(bNamesReg, bNames, branchName, friendName, foundLeaves, leaf, allowDuplicates);
               }
            }

            for (auto leaf : *listOfLeaves) {
               auto castLeaf = static_cast<TLeaf *>(leaf);
               auto leafName = std::string(leaf->GetName());
               auto fullName = branchName + "." + leafName;
               UpdateList(bNamesReg, bNames, fullName, friendName, foundLeaves, castLeaf, allowDuplicates);
            }
         } else {
            // TBranchElement
            // Check if there is explicit or implicit dot in the name

            bool dotIsImplied = false;
            auto be = dynamic_cast<TBranchElement *>(b);
            if (!be)
               throw std::runtime_error("GetBranchNames: unsupported branch type");
            // TClonesArray (3) and STL collection (4)
            if (be->GetType() == 3 || be->GetType() == 4)
               dotIsImplied = true;

            if (dotIsImplied || branchName.back() == '.')
               ExploreBranch(t, bNamesReg, bNames, branch, "", friendName);
            else
               ExploreBranch(t, bNamesReg, bNames, branch, branchName + ".", friendName);

            UpdateList(bNamesReg, bNames, branchName, friendName);
         }
      }
   }

   auto friendTrees = t.GetListOfFriends();

   if (!friendTrees)
      return;

   for (auto friendTreeObj : *friendTrees) {
      auto friendTree = ((TFriendElement *)friendTreeObj)->GetTree();

      std::string frName;
      auto alias = t.GetFriendAlias(friendTree);
      if (alias != nullptr)
         frName = std::string(alias);
      else
         frName = std::string(friendTree->GetName());

      GetBranchNamesImpl(*friendTree, bNamesReg, bNames, analysedTrees, frName, allowDuplicates);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get all the branches names, including the ones of the friend trees
ColumnNames_t GetBranchNames(TTree &t, bool allowDuplicates)
{
   std::set<std::string> bNamesSet;
   ColumnNames_t bNames;
   std::set<TTree *> analysedTrees;
   std::string emptyFrName = "";
   GetBranchNamesImpl(t, bNamesSet, bNames, analysedTrees, emptyFrName, allowDuplicates);
   return bNames;
}

void GetTopLevelBranchNamesImpl(TTree &t, std::set<std::string> &bNamesReg, ColumnNames_t &bNames,
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
      GetTopLevelBranchNamesImpl(*friendTree, bNamesReg, bNames, analysedTrees);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get all the top-level branches names, including the ones of the friend trees
ColumnNames_t GetTopLevelBranchNames(TTree &t)
{
   std::set<std::string> bNamesSet;
   ColumnNames_t bNames;
   std::set<TTree *> analysedTrees;
   GetTopLevelBranchNamesImpl(t, bNamesSet, bNames, analysedTrees);
   return bNames;
}

bool IsValidCppVarName(const std::string &var)
{
   if (var.empty())
      return false;
   const char firstChar = var[0];

   // first character must be either a letter or an underscore
   auto isALetter = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
   const bool isValidFirstChar = firstChar == '_' || isALetter(firstChar);
   if (!isValidFirstChar)
      return false;

   // all characters must be either a letter, an underscore or a number
   auto isANumber = [](char c) { return c >= '0' && c <= '9'; };
   auto isValidTok = [&isALetter, &isANumber](char c) { return c == '_' || isALetter(c) || isANumber(c); };
   for (const char c : var)
      if (!isValidTok(c))
         return false;
      
   return true;
}

void CheckCustomColumn(std::string_view definedCol, TTree *treePtr, const ColumnNames_t &customCols,
                       const ColumnNames_t &dataSourceColumns)
{
   const std::string definedColStr(definedCol);

   if (!IsValidCppVarName(definedColStr)) {
      const auto msg = "Cannot define column \"" + definedColStr + "\": not a valid C++ variable name.";
      throw std::runtime_error(msg);
   }

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

void CheckTypesAndPars(unsigned int nTemplateParams, unsigned int nColumnNames)
{
   if (nTemplateParams != nColumnNames) {
      std::string err_msg = "The number of template parameters specified is ";
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

ColumnNames_t FindUnknownColumns(const ColumnNames_t &requiredCols, const ColumnNames_t &datasetColumns,
                                 const ColumnNames_t &definedCols, const ColumnNames_t &dataSourceColumns)
{
   ColumnNames_t unknownColumns;
   for (auto &column : requiredCols) {
      const auto isBranch = std::find(datasetColumns.begin(), datasetColumns.end(), column) != datasetColumns.end();
      if (isBranch)
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
   const auto str = colName.data();
   const auto goodPrefix = colName.size() > 3 && // has at least more characters than {r,t}df
                           ('r' == str[0] || 't' == str[0]) && // starts with r or t
                           0 == strncmp("df", str+1, 2); // 2nd and 3rd letters are df
   return goodPrefix && '_' == colName.back(); // also ends with '_'
}

std::vector<std::string> GetFilterNames(const std::shared_ptr<RLoopManager> &loopManager)
{
   return loopManager->GetFiltersNames();
}

// Replace all the occurrences of a string by another string
unsigned int Replace(std::string &s, const std::string what, const std::string withWhat)
{
   size_t idx = 0;
   auto numReplacements = 0U;
   while ((idx = s.find(what, idx)) != std::string::npos) {
      s.replace(idx, what.size(), withWhat);
      idx += withWhat.size();
      numReplacements++;
   }
   return numReplacements;
}

// Match expression against names of branches passed as parameter
// Return vector of names of the branches used in the expression
std::vector<std::string> FindUsedColumnNames(std::string_view expression, const ColumnNames_t &branches,
                                             const ColumnNames_t &customColumns, const ColumnNames_t &dsColumns,
                                             const std::map<std::string, std::string> &aliasMap)
{
   // To help matching the regex
   const std::string paddedExpr = " " + std::string(expression) + " ";
   static const std::string regexBit("[^a-zA-Z0-9_]");
   Ssiz_t matchedLen;

   std::vector<std::string> usedBranches;

   // Check which custom columns match
   for (auto &brName : customColumns) {
      std::string bNameRegexContent = regexBit + brName + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &matchedLen)) {
         usedBranches.emplace_back(brName);
      }
   }

   // Check which tree branches match
   for (auto &brName : branches) {
      // Replace "." with "\." for a correct match of sub-branches/leaves
      auto escapedBrName = brName;
      Replace(escapedBrName, std::string("."), std::string("\\."));
      std::string bNameRegexContent = regexBit + escapedBrName + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &matchedLen)) {
         usedBranches.emplace_back(brName);
      }
   }

   // Check which data-source columns match
   for (auto &col : dsColumns) {
      std::string bNameRegexContent = regexBit + col + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &matchedLen)) {
         // if not already found among the other columns
         if (std::find(usedBranches.begin(), usedBranches.end(), col) == usedBranches.end())
            usedBranches.emplace_back(col);
      }
   }

   // Check which aliases match
   for (auto &alias_colName : aliasMap) {
      auto &alias = alias_colName.first;
      std::string bNameRegexContent = regexBit + alias + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &matchedLen)) {
         // if not already found among the other columns
         if (std::find(usedBranches.begin(), usedBranches.end(), alias) == usedBranches.end())
            usedBranches.emplace_back(alias);
      }
   }

   return usedBranches;
}

// TODO we should also replace other invalid chars, like '[],' and spaces
std::vector<std::string> ReplaceDots(const ColumnNames_t &colNames)
{
   std::vector<std::string> dotlessNames = colNames;
   for (auto &c : dotlessNames) {
      const bool hasDot = c.find_first_of('.') != std::string::npos;
      if (hasDot) {
         std::replace(c.begin(), c.end(), '.', '_');
         c.insert(0u, "__tdf_arg_");
      }
   }
   return dotlessNames;
}

// TODO comment well -- there is a lot going on in this function in terms of side-effects
std::vector<std::string> ColumnTypesAsString(ColumnNames_t &colNames, ColumnNames_t &varNames,
                                             const std::map<std::string, std::string> &aliasMap,
                                             const ColumnNames_t &customColNames, TTree *tree, RDataSource *ds,
                                             std::string &expr, unsigned int namespaceID)
{
   std::vector<std::string> colTypes;
   colTypes.reserve(colNames.size());
   const auto aliasMapEnd = aliasMap.end();

   for (auto c = colNames.begin(), v = varNames.begin(); c != colNames.end();) {
      const auto &colName = *c;

      if (colName.find('.') != std::string::npos) {
         // If the column name contains dots, replace its name in the expression with the corresponding varName
         auto numRepl = Replace(expr, colName, *v);
         if (numRepl == 0) {
            // Discard this column: we could not replace it, although we matched it previously
            // This is because it is a substring of a column we already replaced in the expression
            // e.g. "a.b" is a substring column of "a.b.c"
            c = colNames.erase(c);
            v = varNames.erase(v);
            continue;
         }
      } else {
         // Column name with no dots: check the name is still there
         // it might have only been there as part of a column name with dots, e.g. "a" inside "a.b.c"
         const auto paddedExpr = " " + expr + " ";
         static const std::string noWordChars("[^a-zA-Z0-9_]");
         const auto colNameRxBody = noWordChars + colName + noWordChars;
         TRegexp colNameRegex(colNameRxBody.c_str());
         Ssiz_t matchedLen;
         const auto colStillThere = colNameRegex.Index(paddedExpr.c_str(), &matchedLen) != -1;
         if (!colStillThere) {
            c = colNames.erase(c);
            v = varNames.erase(v);
            continue;
         }
      }

      // Replace the colName with the real one in case colName it's an alias
      // The real name is used to get the type, but the variable name will still be colName
      const auto aliasMapIt = aliasMap.find(colName);
      const auto &realColName = aliasMapEnd == aliasMapIt ? colName : aliasMapIt->second;
      // The map is a const reference, so no operator[]
      const auto isCustomCol =
         std::find(customColNames.begin(), customColNames.end(), realColName) != customColNames.end();
      const auto colTypeName = ColumnName2ColumnTypeName(realColName, namespaceID, tree, ds, isCustomCol);
      colTypes.emplace_back(colTypeName);
      ++c, ++v;
   }

   return colTypes;
}

// Jit expression "in the vacuum", throw if cling exits with an error
// This is to make sure that column names, types and expression string are proper C++
void TryToJitExpression(const std::string &expression, const ColumnNames_t &colNames,
                        const std::vector<std::string> &colTypes, bool hasReturnStmt)
{
   R__ASSERT(colNames.size() == colTypes.size());

   static unsigned int iNs = 0U;
   std::stringstream dummyDecl;
   dummyDecl << "namespace __tdf_" << std::to_string(iNs++) << "{ auto tdf_f = []() {";

   for (auto col = colNames.begin(), type = colTypes.begin(); col != colNames.end(); ++col, ++type) {
      dummyDecl << *type << " " << *col << ";\n";
   }

   // Now that branches are declared as variables, put the body of the
   // lambda in dummyDecl and close scopes of f and namespace __tdf_N
   if (hasReturnStmt)
      dummyDecl << expression << "\n;};}";
   else
      dummyDecl << "return " << expression << "\n;};}";

   // Try to declare the dummy lambda, error out if it does not compile
   if (!gInterpreter->Declare(dummyDecl.str().c_str())) {
      auto msg =
         "Cannot interpret the following expression:\n" + std::string(expression) + "\n\nMake sure it is valid C++.";
      throw std::runtime_error(msg);
   }
}

std::string
BuildLambdaString(const std::string &expr, const ColumnNames_t &vars, const ColumnNames_t &varTypes, bool hasReturnStmt)
{
   R__ASSERT(vars.size() == varTypes.size());

   std::stringstream ss;
   ss << "[](";
   for (auto i = 0u; i < vars.size(); ++i) {
      // We pass by reference to avoid expensive copies
      // It can't be const reference in general, as users might want/need to call non-const methods on the values
      ss << varTypes[i] << "& " << vars[i] << ", ";
   }
   if (!vars.empty())
      ss.seekp(-2, ss.cur);

   if (hasReturnStmt)
      ss << "){\n" << expr << "\n}";
   else
      ss << "){return " << expr << "\n;}";

   return ss.str();
}

std::string PrettyPrintAddr(const void *const addr)
{
   std::stringstream s;
   // Windows-friendly
   s << std::hex << std::showbase << reinterpret_cast<size_t>(addr);
   return s.str();
}

// Jit a string filter expression and jit-and-call this->Filter with the appropriate arguments
// Return pointer to the new functional chain node returned by the call, cast to Long_t

void BookFilterJit(RJittedFilter *jittedFilter, void *prevNodeOnHeap, std::string_view name,
                   std::string_view expression, const std::map<std::string, std::string> &aliasMap,
                   const ColumnNames_t &branches, const RDFInternal::RBookedCustomColumns &customCols, TTree *tree, RDataSource *ds,
                   unsigned int namespaceID)
{
   const auto &dsColumns = ds ? ds->GetColumnNames() : ColumnNames_t{};

   // not const because `ColumnTypesAsStrings` might delete redundant matches and replace variable names
   auto usedBranches = FindUsedColumnNames(expression, branches, customCols.GetNames(), dsColumns, aliasMap);
   auto varNames = ReplaceDots(usedBranches);
   auto dotlessExpr = std::string(expression);
   const auto usedColTypes =
      ColumnTypesAsString(usedBranches, varNames, aliasMap, customCols.GetNames(), tree, ds, dotlessExpr, namespaceID);

   TRegexp re("[^a-zA-Z0-9_]return[^a-zA-Z0-9_]");
   Ssiz_t matchedLen;
   const bool hasReturnStmt = re.Index(dotlessExpr, &matchedLen) != -1;

   TryToJitExpression(dotlessExpr, varNames, usedColTypes, hasReturnStmt);

   const auto filterLambda = BuildLambdaString(dotlessExpr, varNames, usedColTypes, hasReturnStmt);

   const auto jittedFilterAddr = PrettyPrintAddr(jittedFilter);
   const auto prevNodeAddr = PrettyPrintAddr(prevNodeOnHeap);

   // columnsOnHeap is deleted by the jitted call to JitFilterHelper
   ROOT::Internal::RDF::RBookedCustomColumns *columnsOnHeap = new ROOT::Internal::RDF::RBookedCustomColumns(customCols);
   const auto columnsOnHeapAddr = PrettyPrintAddr(columnsOnHeap);

   // Produce code snippet that creates the filter and registers it with the corresponding RJittedFilter
   // Windows requires std::hex << std::showbase << (size_t)pointer to produce notation "0x1234"
   std::stringstream filterInvocation;
   filterInvocation << "ROOT::Internal::RDF::JitFilterHelper(" << filterLambda << ", {";
   for (const auto &brName : usedBranches) {
      // Here we selectively replace the brName with the real column name if it's necessary.
      const auto aliasMapIt = aliasMap.find(brName);
      auto &realBrName = aliasMapIt == aliasMap.end() ? brName : aliasMapIt->second;
      filterInvocation << "\"" << realBrName << "\", ";
   }
   if (!usedBranches.empty())
      filterInvocation.seekp(-2, filterInvocation.cur); // remove the last ",
   filterInvocation << "}, \"" << name << "\", "
                    << "reinterpret_cast<ROOT::Detail::RDF::RJittedFilter*>(" << jittedFilterAddr << "), "
                    << "reinterpret_cast<std::shared_ptr<ROOT::Detail::RDF::RNodeBase>*>(" << prevNodeAddr << "),"
                    << "reinterpret_cast<ROOT::Internal::RDF::RBookedCustomColumns*>(" << columnsOnHeapAddr << ")"
                    << ");";

   jittedFilter->GetLoopManagerUnchecked()->ToJit(filterInvocation.str());
}

// Jit a Define call
void BookDefineJit(std::string_view name, std::string_view expression, RLoopManager &lm, RDataSource *ds,
                   const std::shared_ptr<RJittedCustomColumn> &jittedCustomColumn,
                   const RDFInternal::RBookedCustomColumns &customCols)
{
   const auto &aliasMap = lm.GetAliasMap();
   auto *const tree = lm.GetTree();
   const auto branches = tree ? RDFInternal::GetBranchNames(*tree) : ColumnNames_t();
   const auto namespaceID = lm.GetID();
   const auto &dsColumns = ds ? ds->GetColumnNames() : ColumnNames_t{};

   // not const because `ColumnTypesAsStrings` might delete redundant matches and replace variable names
   auto usedBranches = FindUsedColumnNames(expression, branches, customCols.GetNames(), dsColumns, aliasMap);
   auto varNames = ReplaceDots(usedBranches);
   auto dotlessExpr = std::string(expression);
   const auto usedColTypes =
      ColumnTypesAsString(usedBranches, varNames, aliasMap, customCols.GetNames(), tree, ds, dotlessExpr, namespaceID);

   TRegexp re("[^a-zA-Z0-9_]return[^a-zA-Z0-9_]");
   Ssiz_t matchedLen;
   const bool hasReturnStmt = re.Index(dotlessExpr, &matchedLen) != -1;

   TryToJitExpression(dotlessExpr, varNames, usedColTypes, hasReturnStmt);

  const auto definelambda = BuildLambdaString(dotlessExpr, varNames, usedColTypes, hasReturnStmt);
  const auto lambdaName = "eval_" + std::string(name);
  const auto ns = "__tdf" + std::to_string(namespaceID);

   auto customColumnsCopy = new RDFInternal::RBookedCustomColumns(customCols);
   auto customColumnsAddr = PrettyPrintAddr(customColumnsCopy);

   // Declare the lambda variable and an alias for the type of the defined column in namespace __tdf
   // This assumes that a given variable is Define'd once per RDataFrame -- we might want to relax this requirement
   // to let python users execute a Define cell multiple times
   const auto defineDeclaration =
      "namespace " + ns + " { auto " + lambdaName + " = " + definelambda + ";\n" + "using " + std::string(name) +
      "_type = typename ROOT::TypeTraits::CallableTraits<decltype(" + lambdaName + " )>::ret_type;  }\n";
   gInterpreter->Declare(defineDeclaration.c_str());

   std::stringstream defineInvocation;
   defineInvocation << "ROOT::Internal::RDF::JitDefineHelper(" << definelambda << ", {";
   for (auto brName : usedBranches) {
      // Here we selectively replace the brName with the real column name if it's necessary.
      auto aliasMapIt = aliasMap.find(brName);
      auto &realBrName = aliasMapIt == aliasMap.end() ? brName : aliasMapIt->second;
      defineInvocation << "\"" << realBrName << "\", ";
   }
   if (!usedBranches.empty())
      defineInvocation.seekp(-2, defineInvocation.cur); // remove the last ",
   defineInvocation << "}, \"" << name << "\", reinterpret_cast<ROOT::Detail::RDF::RLoopManager*>("
                    << PrettyPrintAddr(&lm) << "), *reinterpret_cast<ROOT::Detail::RDF::RJittedCustomColumn*>("
                    << PrettyPrintAddr(jittedCustomColumn.get()) << "),"
                    << "reinterpret_cast<ROOT::Internal::RDF::RBookedCustomColumns*>(" << customColumnsAddr << ")"
                    << ");";

   lm.ToJit(defineInvocation.str());

}

// Jit and call something equivalent to "this->BuildAndBook<BranchTypes...>(params...)"
// (see comments in the body for actual jitted code)
std::string JitBuildAction(const ColumnNames_t &bl, void *prevNode, const std::type_info &art, const std::type_info &at,
                           void *rOnHeap, TTree *tree, const unsigned int nSlots,
                           const RDFInternal::RBookedCustomColumns &customCols, RDataSource *ds,
                           std::shared_ptr<RJittedAction> *jittedActionOnHeap, unsigned int namespaceID)
{
   auto nBranches = bl.size();

   // retrieve branch type names as strings
   std::vector<std::string> columnTypeNames(nBranches);
   for (auto i = 0u; i < nBranches; ++i) {
      const auto isCustomCol = customCols.HasName(bl[i]);
      const auto columnTypeName = ColumnName2ColumnTypeName(bl[i], namespaceID, tree, ds, isCustomCol);
      if (columnTypeName.empty()) {
         std::string exceptionText = "The type of column ";
         exceptionText += bl[i];
         exceptionText += " could not be guessed. Please specify one.";
         throw std::runtime_error(exceptionText.c_str());
      }
      columnTypeNames[i] = columnTypeName;
   }

   // retrieve type of result of the action as a string
   auto actionResultTypeClass = TClass::GetClass(art);
   if (!actionResultTypeClass) {
      std::string exceptionText = "An error occurred while inferring the result type of an operation.";
      throw std::runtime_error(exceptionText.c_str());
   }
   const auto actionResultTypeName = actionResultTypeClass->GetName();

   // retrieve type of action as a string
   auto actionTypeClass = TClass::GetClass(at);
   if (!actionTypeClass) {
      std::string exceptionText = "An error occurred while inferring the action type of the operation.";
      throw std::runtime_error(exceptionText.c_str());
   }
   const auto actionTypeName = actionTypeClass->GetName();

   auto customColumnsCopy = new RDFInternal::RBookedCustomColumns(customCols); // deleted in jitted CallBuildAction
   auto customColumnsAddr = PrettyPrintAddr(customColumnsCopy);

   // Build a call to CallBuildAction with the appropriate argument. When run through the interpreter, this code will
   // just-in-time create an RAction object and it will assign it to its corresponding RJittedAction.
   std::stringstream createAction_str;
   createAction_str << "ROOT::Internal::RDF::CallBuildAction"
                    << "<" << actionTypeName;
   for (auto &colType : columnTypeNames)
      createAction_str << ", " << colType;
   // on Windows, to prefix the hexadecimal value of a pointer with '0x',
   // one need to write: std::hex << std::showbase << (size_t)pointer
   createAction_str << ">(reinterpret_cast<std::shared_ptr<ROOT::Detail::RDF::RNodeBase>*>("
                    << PrettyPrintAddr(prevNode) << "), {";
   for (auto i = 0u; i < bl.size(); ++i) {
      if (i != 0u)
         createAction_str << ", ";
      createAction_str << '"' << bl[i] << '"';
   }
   createAction_str << "}, " << std::dec << std::noshowbase << nSlots << ", reinterpret_cast<" << actionResultTypeName
                    << "*>(" << PrettyPrintAddr(rOnHeap) << ")"
                    << ", reinterpret_cast<std::shared_ptr<ROOT::Internal::RDF::RJittedAction>*>("
                    << PrettyPrintAddr(jittedActionOnHeap) << "),"
                    << "reinterpret_cast<ROOT::Internal::RDF::RBookedCustomColumns*>(" << customColumnsAddr << ")"
                    << ");";
   return createAction_str.str();
}

bool AtLeastOneEmptyString(const std::vector<std::string_view> strings)
{
   for (const auto &s : strings) {
      if (s.empty())
         return true;
   }
   return false;
}

std::shared_ptr<RNodeBase> UpcastNode(std::shared_ptr<RNodeBase> ptr) { return ptr; }

/// Given the desired number of columns and the user-provided list of columns:
/// * fallback to using the first nColumns default columns if needed (or throw if nColumns > nDefaultColumns)
/// * check that selected column names refer to valid branches, custom columns or datasource columns (throw if not)
/// Return the list of selected column names.
ColumnNames_t GetValidatedColumnNames(RLoopManager &lm, const unsigned int nColumns, const ColumnNames_t &columns,
                                      const ColumnNames_t &datasetColumns, const ColumnNames_t &validCustomColumns,
                                      RDataSource *ds)
{
   const auto &defaultColumns = lm.GetDefaultColumnNames();
   auto selectedColumns = SelectColumns(nColumns, columns, defaultColumns);
   const auto unknownColumns = FindUnknownColumns(selectedColumns, datasetColumns, validCustomColumns,
                                                  ds ? ds->GetColumnNames() : ColumnNames_t{});

   if (!unknownColumns.empty()) {
      // throw
      std::stringstream unknowns;
      std::string delim = unknownColumns.size() > 1 ? "s: " : ": "; // singular/plural
      for (auto &unknownColumn : unknownColumns) {
         unknowns << delim << unknownColumn;
         delim = ',';
      }
      throw std::runtime_error("Unknown column" + unknowns.str());
   }

   // Now we need to check within the aliases if some of the yet unknown names can be recovered
   auto &aliasMap = lm.GetAliasMap();
   auto aliasMapEnd = aliasMap.end();

   for (auto idx : ROOT::TSeqU(selectedColumns.size())) {
      const auto &colName = selectedColumns[idx];
      const auto aliasColumnNameIt = aliasMap.find(colName);
      if (aliasMapEnd != aliasColumnNameIt) {
         selectedColumns[idx] = aliasColumnNameIt->second;
      }
   }

   return selectedColumns;
}

/// Return a bitset each element of which indicates whether the corresponding element in `selectedColumns` is the
/// name of a column that must be defined via datasource. All elements of the returned vector are false if no
/// data-source is present.
std::vector<bool> FindUndefinedDSColumns(const ColumnNames_t &requestedCols, const ColumnNames_t &definedCols)
{
   const auto nColumns = requestedCols.size();
   std::vector<bool> mustBeDefined(nColumns, false);
   for (auto i = 0u; i < nColumns; ++i)
      mustBeDefined[i] = std::find(definedCols.begin(), definedCols.end(), requestedCols[i]) == definedCols.end();
   return mustBeDefined;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
