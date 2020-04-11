// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/InterfaceUtils.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/TSeq.hxx>
#include <RtypesCore.h>
#include <TDirectory.h>
#include <TChain.h>
#include <TClass.h>
#include <TClassEdit.h>
#include <TFriendElement.h>
#include <TInterpreter.h>
#include <TObject.h>
#include <TRegexp.h>
#include <TPRegexp.h>
#include <TString.h>
#include <TTree.h>

// pragma to disable warnings on Rcpp which have
// so many noise compiling
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "lexertk.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <set>
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

/// A tokeniser for the expression which is in C++
/// The goal is to extract all names which are potentially
/// columns. The difficulty is to catch also the names containing dots.
std::set<std::string> GetPotentialColumnNames(const std::string &expr)
{
   lexertk::generator generator;
   const auto ok = generator.process(expr);
   if (!ok) {
      const auto msg = "Failed to tokenize expression:\n" + expr + "\n\nMake sure it is valid C++.";
      throw std::runtime_error(msg);
   }

   std::set<std::string> potCols;
   const auto nToks = generator.size();
   std::string potColWithDots;

   auto IsSymbol = [](const lexertk::token &t) { return t.type == lexertk::token::e_symbol; };
   auto IsDot = [](const lexertk::token &t) { return t.value == "."; };

   // Now we start iterating over the tokens
   for (auto i = 0ULL; i < nToks; ++i) {
      auto &tok = generator[i];
      if (!IsSymbol(tok))
         continue;

      if (i == 0 || (i > 0 && !IsDot(generator[i - 1])))
         potCols.insert(tok.value);

      // after the current token we may have a chain of .<symbol>.<symbol>...
      // we need to build a set of potential columns incrementally
      // and stop at the right point. All this advancing the token
      // cursor.
      potColWithDots = tok.value;
      while (i < nToks) {
         if (i + 2 == nToks)
            break;
         auto &nextTok = generator[i + 1];
         auto &next2nextTok = generator[i + 2];
         if (!IsDot(nextTok) || !IsSymbol(next2nextTok)) {
            break;
         }
         potColWithDots += "." + next2nextTok.value;
         potCols.insert(potColWithDots);
         i += 2;
      }
      potColWithDots = "";
   }
   return potCols;
}

// The set here is used as a registry, the real list, which keeps the order, is
// the one in the vector
class RActionBase;

HeadNode_t CreateSnapshotRDF(const ColumnNames_t &validCols,
                            std::string_view treeName,
                            std::string_view fileName,
                            bool isLazy,
                            RLoopManager &loopManager,
                            std::unique_ptr<RDFInternal::RActionBase> actionPtr)
{
   // create new RDF
   ::TDirectory::TContext ctxt;
   auto snapshotRDF = std::make_shared<ROOT::RDataFrame>(treeName, fileName, validCols);
   auto snapshotRDFResPtr = MakeResultPtr(snapshotRDF, loopManager, std::move(actionPtr));

   if (!isLazy) {
      *snapshotRDFResPtr;
   }
   return snapshotRDFResPtr;
}

std::string DemangleTypeIdName(const std::type_info &typeInfo)
{
   int dummy(0);
   return TClassEdit::DemangleTypeIdName(typeInfo, dummy);
}

ColumnNames_t ConvertRegexToColumns(const RDFInternal::RBookedCustomColumns & customColumns,
                                    TTree *tree,
                                    ROOT::RDF::RDataSource *dataSource,
                                    std::string_view columnNameRegexp,
                                    std::string_view callerName)
{
   const auto theRegexSize = columnNameRegexp.size();
   std::string theRegex(columnNameRegexp);

   const auto isEmptyRegex = 0 == theRegexSize;
   // This is to avoid cases where branches called b1, b2, b3 are all matched by expression "b"
   if (theRegexSize > 0 && theRegex[0] != '^')
      theRegex = "^" + theRegex;
   if (theRegexSize > 0 && theRegex[theRegexSize - 1] != '$')
      theRegex = theRegex + "$";

   ColumnNames_t selectedColumns;
   selectedColumns.reserve(32);

   // Since we support gcc48 and it does not provide in its stl std::regex,
   // we need to use TRegexp
   TPRegexp regexp(theRegex);
   for (auto &&branchName : customColumns.GetNames()) {
      if ((isEmptyRegex || 0 != regexp.Match(branchName.c_str())) &&
            !RDFInternal::IsInternalColumn(branchName)) {
         selectedColumns.emplace_back(branchName);
      }
   }

   if (tree) {
      auto branchNames = RDFInternal::GetTopLevelBranchNames(*tree);
      for (auto &branchName : branchNames) {
         if (isEmptyRegex || 0 != regexp.Match(branchName.c_str())) {
            selectedColumns.emplace_back(branchName);
         }
      }
   }

   if (dataSource) {
      auto &dsColNames = dataSource->GetColumnNames();
      for (auto &dsColName : dsColNames) {
         if ((isEmptyRegex || 0 != regexp.Match(dsColName.c_str())) &&
               !RDFInternal::IsInternalColumn(dsColName)) {
            selectedColumns.emplace_back(dsColName);
         }
      }
   }

   if (selectedColumns.empty()) {
      std::string text(callerName);
      if (columnNameRegexp.empty()) {
         text = ": there is no column available to match.";
      } else {
         text = ": regex \"" + std::string(columnNameRegexp) + "\" did not match any column.";
      }
      throw std::runtime_error(text);
   }
   return selectedColumns;
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
                       const std::map<std::string, std::string> &aliasMap, const ColumnNames_t &dataSourceColumns)
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

   // Check if the definedCol is an alias
   const auto aliasColNameIt = aliasMap.find(definedColStr);
   if (aliasColNameIt != aliasMap.end()) {
      const auto msg = "An alias with name " + definedColStr + " pointing to column " +
      aliasColNameIt->second + " is already existing.";
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
   const auto goodPrefix = colName.size() > 3 &&               // has at least more characters than {r,t}df
                           ('r' == str[0] || 't' == str[0]) && // starts with r or t
                           0 == strncmp("df", str + 1, 2);     // 2nd and 3rd letters are df
   return goodPrefix && '_' == colName.back();                 // also ends with '_'
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
std::vector<std::string> FindUsedColumnNames(std::string_view expression, ColumnNames_t branches,
                                             const ColumnNames_t &customColumns, const ColumnNames_t &dsColumns,
                                             const std::map<std::string, std::string> &aliasMap)
{
   // To help matching the regex
   const auto potCols = GetPotentialColumnNames(std::string(expression));

   if (potCols.size() == 0) return {};

   std::set<std::string> usedBranches;

   // Check which custom columns match
   for (auto &brName : customColumns) {
      if (potCols.find(brName) != potCols.end()) {
         usedBranches.insert(brName);
      }
   }

   // Check which tree branches match
   // We need to match the longest

   // First: reverse sort to have longer branches before, e.g.
   // a.b.c
   // a.b
   // a
   // We want that the longest branch ends up in usedBranches before.
   std::sort(branches.begin(), branches.end(),
             [](const std::string &s0, const std::string &s1) {return s0 > s1;});

   for (auto &brName : branches) {
      // If the branch is not in the potential columns, we simply move on
      if (potCols.find(brName) == potCols.end()) {
         continue;
      }
      // If not, we check if the branch name is contained in one of the branch
      // names which we already added to the usedBranches.
      auto isContained = [&brName](const std::string &usedBr) {
         // We check two things:
         // 1. That the string is contained, e.g. a.b is contained in a.b.c.d
         // 2. That the number of '.'s is greater, otherwise in situations where
         //    2 branches have names like br0 and br01, br0 is not matched (ROOT-9929)
         return usedBr.find(brName) != std::string::npos &&
           std::count(usedBr.begin(), usedBr.end(), '.') > std::count(brName.begin(), brName.end(), '.');
         };
      auto it = std::find_if(usedBranches.begin(), usedBranches.end(), isContained);
      if (it == usedBranches.end()) {
         usedBranches.insert(brName);
      }
   }

   // Check which data-source columns match
   for (auto &col : dsColumns) {
      if (potCols.find(col) != potCols.end()) {
         usedBranches.insert(col);
      }
   }

   // Check which aliases match
   for (auto &alias_colName : aliasMap) {
      auto &alias = alias_colName.first;
      if (potCols.find(alias) != potCols.end()) {
         usedBranches.insert(alias);
      }
   }

   return std::vector<std::string>(usedBranches.begin(), usedBranches.end());
}

// TODO we should also replace other invalid chars, like '[],' and spaces
std::vector<std::string> ReplaceDots(const ColumnNames_t &colNames)
{
   std::vector<std::string> dotlessNames = colNames;
   for (auto &c : dotlessNames) {
      const bool hasDot = c.find_first_of('.') != std::string::npos;
      if (hasDot) {
         std::replace(c.begin(), c.end(), '.', '_');
         c.insert(0u, "__rdf_arg_");
      }
   }
   return dotlessNames;
}

// TODO comment well -- there is a lot going on in this function in terms of side-effects
std::vector<std::string> ColumnTypesAsString(ColumnNames_t &colNames, ColumnNames_t &varNames,
                                             const std::map<std::string, std::string> &aliasMap, TTree *tree,
                                             RDataSource *ds, std::string &expr, unsigned int namespaceID,
                                             const RDFInternal::RBookedCustomColumns &customCols)
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
      const auto isCustomCol = customCols.HasName(realColName);
      const auto customColID = isCustomCol ? customCols.GetColumns().at(realColName)->GetID() : 0;
      const auto colTypeName =
         ColumnName2ColumnTypeName(realColName, namespaceID, tree, ds, isCustomCol, /*vector2rvec=*/true, customColID);
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
   dummyDecl << "namespace __rdf_" << std::to_string(iNs++) << "{ auto rdf_f = []() {";

   for (auto col = colNames.begin(), type = colTypes.begin(); col != colNames.end(); ++col, ++type) {
      dummyDecl << *type << " " << *col << ";\n";
   }

   // Now that branches are declared as variables, put the body of the
   // lambda in dummyDecl and close scopes of f and namespace __rdf_N
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
      ss << "){";
   else
      ss << "){return ";
   ss << expr << "\n;}";

   return ss.str();
}

std::string PrettyPrintAddr(const void *const addr)
{
   std::stringstream s;
   // Windows-friendly
   s << std::hex << std::showbase << reinterpret_cast<size_t>(addr);
   return s.str();
}

void BookFilterJit(RJittedFilter *jittedFilter, void *prevNodeOnHeap, std::string_view name,
                   std::string_view expression, const std::map<std::string, std::string> &aliasMap,
                   const ColumnNames_t &branches, const RDFInternal::RBookedCustomColumns &customCols, TTree *tree,
                   RDataSource *ds, unsigned int namespaceID)
{
   const auto &dsColumns = ds ? ds->GetColumnNames() : ColumnNames_t{};

   // not const because `ColumnTypesAsStrings` might delete redundant matches and replace variable names
   auto usedBranches = FindUsedColumnNames(expression, branches, customCols.GetNames(), dsColumns, aliasMap);
   auto varNames = ReplaceDots(usedBranches);
   auto dotlessExpr = std::string(expression);
   const auto usedColTypes =
      ColumnTypesAsString(usedBranches, varNames, aliasMap, tree, ds, dotlessExpr, namespaceID, customCols);

   TRegexp re("[^a-zA-Z0-9_]?return[^a-zA-Z0-9_]");
   Ssiz_t matchedLen;
   const bool hasReturnStmt = re.Index(dotlessExpr, &matchedLen) != -1;

   auto lm = jittedFilter->GetLoopManagerUnchecked();
   lm->JitDeclarations(); // TryToJitExpression might need some of the Define'd column type aliases
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
                    << ");\n";

   lm->ToJitExec(filterInvocation.str());
}

// Jit a Define call
void BookDefineJit(std::string_view name, std::string_view expression, RLoopManager &lm, RDataSource *ds,
                   const std::shared_ptr<RJittedCustomColumn> &jittedCustomColumn,
                   const RDFInternal::RBookedCustomColumns &customCols, const ColumnNames_t &branches)
{
   const auto &aliasMap = lm.GetAliasMap();
   auto *const tree = lm.GetTree();
   const auto namespaceID = lm.GetID();
   const auto &dsColumns = ds ? ds->GetColumnNames() : ColumnNames_t{};

   // not const because `ColumnTypesAsStrings` might delete redundant matches and replace variable names
   auto usedBranches = FindUsedColumnNames(expression, branches, customCols.GetNames(), dsColumns, aliasMap);
   auto varNames = ReplaceDots(usedBranches);
   auto dotlessExpr = std::string(expression);
   const auto usedColTypes =
      ColumnTypesAsString(usedBranches, varNames, aliasMap, tree, ds, dotlessExpr, namespaceID, customCols);

   TRegexp re("[^a-zA-Z0-9_]?return[^a-zA-Z0-9_]");
   Ssiz_t matchedLen;
   const bool hasReturnStmt = re.Index(dotlessExpr, &matchedLen) != -1;

   lm.JitDeclarations(); // TryToJitExpression might need some of the Define'd column type aliases
   TryToJitExpression(dotlessExpr, varNames, usedColTypes, hasReturnStmt);

   const auto definelambda = BuildLambdaString(dotlessExpr, varNames, usedColTypes, hasReturnStmt);
   const auto customColID = std::to_string(jittedCustomColumn->GetID());
   const auto lambdaName = "eval_" + std::string(name) + customColID;
   const auto ns = "__rdf" + std::to_string(namespaceID);

   auto customColumnsCopy = new RDFInternal::RBookedCustomColumns(customCols);
   auto customColumnsAddr = PrettyPrintAddr(customColumnsCopy);

   // Declare the lambda variable and an alias for the type of the defined column in namespace __rdf
   // This assumes that a given variable is Define'd once per RDataFrame -- we might want to relax this requirement
   // to let python users execute a Define cell multiple times
   const auto defineDeclaration =
      "namespace " + ns + " { auto " + lambdaName + " = " + definelambda + ";\n" + "using " + std::string(name) +
      customColID + "_type = typename ROOT::TypeTraits::CallableTraits<decltype(" + lambdaName + " )>::ret_type;  }\n";
   lm.ToJitDeclare(defineDeclaration);

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
                    << ");\n";

   lm.ToJitExec(defineInvocation.str());
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
      const auto customColID = isCustomCol ? customCols.GetColumns().at(bl[i])->GetID() : 0;
      const auto columnTypeName =
         ColumnName2ColumnTypeName(bl[i], namespaceID, tree, ds, isCustomCol, /*vector2rvec=*/true, customColID);
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

std::shared_ptr<RNodeBase> UpcastNode(std::shared_ptr<RNodeBase> ptr)
{
   return ptr;
}

/// Given the desired number of columns and the user-provided list of columns:
/// * fallback to using the first nColumns default columns if needed (or throw if nColumns > nDefaultColumns)
/// * check that selected column names refer to valid branches, custom columns or datasource columns (throw if not)
/// * replace column names from aliases by the actual column name
/// Return the list of selected column names.
ColumnNames_t GetValidatedColumnNames(RLoopManager &lm, const unsigned int nColumns, const ColumnNames_t &columns,
                                      const ColumnNames_t &validCustomColumns, RDataSource *ds)
{
   const auto &defaultColumns = lm.GetDefaultColumnNames();
   auto selectedColumns = SelectColumns(nColumns, columns, defaultColumns);
   const auto &validBranchNames = lm.GetBranchNames();
   const auto unknownColumns = FindUnknownColumns(selectedColumns, validBranchNames, validCustomColumns,
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
