// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TDFInterfaceUtils.hxx>
#include <ROOT/RStringView.hxx>
#include <RtypesCore.h>
#include <TClass.h>
#include <TFriendElement.h>
#include <TInterpreter.h>
#include <TObject.h>
#include <TRegexp.h>
#include <TString.h>
#include <TTree.h>

#include <iosfwd>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace ROOT {
namespace Detail {
namespace TDF {
class TCustomColumnBase;
class TFilterBase;
class TLoopManager;
class TRangeBase;
}  // namespace TDF
}  // namespace Detail
namespace Experimental {
namespace TDF {
class TDataSource;
}  // namespace TDF
}  // namespace Experimental
}  // namespace ROOT

namespace ROOT {
namespace Internal {
namespace TDF {

// The set here is used as a registry, the real list, which keeps the order, is
// the one in the vector
class TActionBase;

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

bool IsTreeLeaf(TTree &t, const std::string &leaf)
{
   // TODO understand why GetBranch is also needed (run the tests without, inspect failures)
   if (t.GetBranch(leaf.c_str()) != nullptr)
      return true;
   if (t.GetLeaf(leaf.c_str()) != nullptr)
      return true;
   auto lastDot = leaf.find_last_of('.');
   if (lastDot != std::string::npos) {
      std::string leafWithSlash(leaf);
      leafWithSlash[lastDot] = '/';
      if (t.GetLeaf(leafWithSlash.c_str()) != nullptr)
         return true;
   }

   return false;
}

ColumnNames_t FindUnknownColumns(const ColumnNames_t &requiredCols, TTree *tree, const ColumnNames_t &definedCols,
                                 const ColumnNames_t &dataSourceColumns)
{
   ColumnNames_t unknownColumns;
   for (auto &column : requiredCols) {
      if (tree != nullptr && IsTreeLeaf(*tree, column))
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

// Match expression against names of branches passed as parameter
// Return vector of names of the branches used in the expression
std::vector<std::string> FindUsedColumnNames(std::string_view expression, const ColumnNames_t &branches,
                                             const ColumnNames_t &customColumns, const ColumnNames_t &dsColumns,
                                             const std::map<std::string, std::string> &aliasMap)
{
   // To help matching the regex
   const std::string paddedExpr = " " + std::string(expression) + " ";
   int paddedExprLen = paddedExpr.size();
   static const std::string regexBit("[^a-zA-Z0-9_]");

   std::vector<std::string> usedBranches;

   // Check which custom columns match
   for (auto &brName : customColumns) {
      std::string bNameRegexContent = regexBit + brName + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
         usedBranches.emplace_back(brName);
      }
   }

   // Check which tree branches match
   for (auto &brName : branches) {
      std::string bNameRegexContent = regexBit + brName + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
         usedBranches.emplace_back(brName);
      }
   }

   // Check which data-source columns match
   for (auto &col : dsColumns) {
      std::string bNameRegexContent = regexBit + col + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
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
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
         // if not already found among the other columns
         if (std::find(usedBranches.begin(), usedBranches.end(), alias) == usedBranches.end())
            usedBranches.emplace_back(alias);
      }
   }

   return usedBranches;
}

// Jit a string filter or a string temporary column, call this->Define or this->Filter as needed
// Return pointer to the new functional chain node returned by the call, cast to Long_t
Long_t JitTransformation(void *thisPtr, std::string_view methodName, std::string_view interfaceTypeName,
                         std::string_view name, std::string_view expression,
                         const std::map<std::string, std::string> &aliasMap, const ColumnNames_t &branches,
                         const std::vector<std::string> &customColumns,
                         const std::map<std::string, TmpBranchBasePtr_t> &tmpBookedBranches, TTree *tree,
                         std::string_view returnTypeName, TDataSource *ds)
{
   const auto &dsColumns = ds ? ds->GetColumnNames() : ColumnNames_t{};
   auto usedBranches = FindUsedColumnNames(expression, branches, customColumns, dsColumns, aliasMap);
   auto exprNeedsVariables = !usedBranches.empty();

   // Move to the preparation of the jitting
   // We put all of the jitted entities in function f in namespace __tdf_N, where N is a monotonically increasing index
   // and then try to declare that function to make sure column names, types and expression are proper C++
   std::vector<std::string> usedBranchesTypes;
   static unsigned int iNs = 0U;
   std::stringstream dummyDecl;
   dummyDecl << "namespace __tdf_" << std::to_string(iNs++) << "{ auto __tdf_lambda = []() {";

   // Declare variables with the same name as the column used by this transformation
   auto aliasMapEnd = aliasMap.end();
   if (exprNeedsVariables) {
      for (auto &brName : usedBranches) {
         // Here we replace on the fly the brName with the real one in case brName it's an alias
         // This is then used to get the type. The variable name will be brName;
         auto aliasMapIt = aliasMap.find(brName);
         auto &realBrName = aliasMapEnd == aliasMapIt ? brName : aliasMapIt->second;
         // The map is a const reference, so no operator[]
         auto tmpBrIt = tmpBookedBranches.find(realBrName);
         auto tmpBr = tmpBrIt == tmpBookedBranches.end() ? nullptr : tmpBrIt->second.get();
         auto brTypeName = ColumnName2ColumnTypeName(realBrName, tree, tmpBr, ds);
         dummyDecl << brTypeName << " " << brName << ";\n";
         usedBranchesTypes.emplace_back(brTypeName);
      }
   }

   TRegexp re("[^a-zA-Z0-9_]return[^a-zA-Z0-9_]");
   int exprSize = expression.size();
   bool hasReturnStmt = re.Index(std::string(expression), &exprSize) != -1;

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

   // Now we build the lambda and we invoke the method with it in the jitted world
   std::stringstream ss;
   ss << "[](";
   for (unsigned int i = 0; i < usedBranchesTypes.size(); ++i) {
      // We pass by reference to avoid expensive copies
      // It can't be const reference in general, as users might want/need to call non-const methods on the values
      // Here we do not replace anything: the name of the parameters of the lambda does not need to be the real
      // column name, and sometimes it has to be an alias to compile (e.g. "b_a" as alias for "b.a")
      ss << usedBranchesTypes[i] << "& " << usedBranches[i] << ", ";
   }
   if (!usedBranchesTypes.empty())
      ss.seekp(-2, ss.cur);

   if (hasReturnStmt)
      ss << "){\n" << expression << "\n}";
   else
      ss << "){return " << expression << "\n;}";

   auto filterLambda = ss.str();

   // The TInterface type to convert the result to. For example, Filter returns a TInterface<TFilter<F,P>> but when
   // returning it from a jitted call we need to convert it to TInterface<TFilterBase> as we are missing information
   // on types F and P at compile time.
   const auto targetTypeName = "ROOT::Experimental::TDF::TInterface<" + std::string(returnTypeName) + ">";

   // Here we have two cases: filter and column
   ss.str("");
   // on Windows, to prefix the hexadecimal value of a pointer with '0x',
   // one need to write: std::hex << std::showbase << (size_t)pointer
   ss << targetTypeName << "(((" << interfaceTypeName << "*)" << std::hex << std::showbase << (size_t)thisPtr << ")->"
      << methodName << "(";
   if (methodName == "Define") {
      ss << "\"" << name << "\", ";
   }
   ss << filterLambda << ", {";
   for (auto brName : usedBranches) {
      // Here we selectively replace the brName with the real column name if it's necessary.
      auto aliasMapIt = aliasMap.find(brName);
      auto &realBrName = aliasMapEnd == aliasMapIt ? brName : aliasMapIt->second;
      ss << "\"" << realBrName << "\", ";
   }
   if (exprNeedsVariables)
      ss.seekp(-2, ss.cur); // remove the last ",
   ss << "}";

   if (methodName == "Filter") {
      ss << ", \"" << name << "\"";
   }

   ss << "));";

   TInterpreter::EErrorCode interpErrCode;
   auto retVal = gInterpreter->Calc(ss.str().c_str(), &interpErrCode);
   if (TInterpreter::EErrorCode::kNoError != interpErrCode || !retVal) {
      std::string msg = "Cannot interpret the invocation to " + std::string(methodName) + ":  ";
      msg += ss.str();
      if (TInterpreter::EErrorCode::kNoError != interpErrCode) {
         msg += "\nInterpreter error code is " + std::to_string(interpErrCode) + ".";
      }
      throw std::runtime_error(msg);
   }
   return retVal;
}

// Jit and call something equivalent to "this->BuildAndBook<BranchTypes...>(params...)"
// (see comments in the body for actual jitted code)
std::string JitBuildAndBook(const ColumnNames_t &bl, const std::string &prevNodeTypename, void *prevNode,
                            const std::type_info &art, const std::type_info &at, const void *rOnHeap, TTree *tree,
                            const unsigned int nSlots, const std::map<std::string, TmpBranchBasePtr_t> &customColumns,
                            TDataSource *ds, const std::shared_ptr<TActionBase *> *const actionPtrPtr)
{
   auto nBranches = bl.size();

   // retrieve pointers to temporary columns (null if the column is not temporary)
   std::vector<TCustomColumnBase *> tmpBranchPtrs(nBranches, nullptr);
   for (auto i = 0u; i < nBranches; ++i) {
      auto tmpBranchIt = customColumns.find(bl[i]);
      if (tmpBranchIt != customColumns.end())
         tmpBranchPtrs[i] = tmpBranchIt->second.get();
   }

   // retrieve branch type names as strings
   std::vector<std::string> columnTypeNames(nBranches);
   for (auto i = 0u; i < nBranches; ++i) {
      const auto columnTypeName = ColumnName2ColumnTypeName(bl[i], tree, tmpBranchPtrs[i], ds);
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

   // createAction_str will contain the following:
   // ROOT::Internal::TDF::CallBuildAndBook<actionType, branchType1, branchType2...>(
   //   *reinterpret_cast<PrevNodeType*>(prevNode), { bl[0], bl[1], ... }, reinterpret_cast<actionResultType*>(rOnHeap),
   //   reinterpret_cast<shared_ptr<TActionBase*>*>(actionPtrPtr))
   std::stringstream createAction_str;
   createAction_str << "ROOT::Internal::TDF::CallBuildAndBook"
                    << "<" << actionTypeName;
   for (auto &colType : columnTypeNames)
      createAction_str << ", " << colType;
   // on Windows, to prefix the hexadecimal value of a pointer with '0x',
   // one need to write: std::hex << std::showbase << (size_t)pointer
   createAction_str << ">(*reinterpret_cast<" << prevNodeTypename << "*>(" << std::hex << std::showbase
                    << (size_t)prevNode << "), {";
   for (auto i = 0u; i < bl.size(); ++i) {
      if (i != 0u)
         createAction_str << ", ";
      createAction_str << '"' << bl[i] << '"';
   }
   createAction_str << "}, " << std::dec << std::noshowbase << nSlots << ", reinterpret_cast<" << actionResultTypeName
                    << "*>(" << std::hex << std::showbase << (size_t)rOnHeap << ")"
                    << ", reinterpret_cast<const std::shared_ptr<ROOT::Internal::TDF::TActionBase*>*>(" << std::hex
                    << std::showbase << (size_t)actionPtrPtr << "));";
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

/*** Take a shared_ptr<Node<T1,T2,...>> and return a shared_ptr<NodeBase> ***/
std::shared_ptr<TFilterBase> UpcastNode(const std::shared_ptr<TFilterBase> ptr)
{
   return ptr;
}

std::shared_ptr<TCustomColumnBase> UpcastNode(const std::shared_ptr<TCustomColumnBase> ptr)
{
   return ptr;
}

std::shared_ptr<TRangeBase> UpcastNode(const std::shared_ptr<TRangeBase> ptr)
{
   return ptr;
}

std::shared_ptr<TLoopManager> UpcastNode(const std::shared_ptr<TLoopManager> ptr)
{
   return ptr;
}
/****************************************************************************/

/// Given the desired number of columns and the user-provided list of columns:
/// * fallback to using the first nColumns default columns if needed (or throw if nColumns > nDefaultColumns)
/// * check that selected column names refer to valid branches, custom columns or datasource columns (throw if not)
/// Return the list of selected column names.
ColumnNames_t GetValidatedColumnNames(TLoopManager &lm, const unsigned int nColumns, const ColumnNames_t &columns,
                                      const ColumnNames_t &validCustomColumns, TDataSource *ds)
{
   const auto &defaultColumns = lm.GetDefaultColumnNames();
   auto selectedColumns = SelectColumns(nColumns, columns, defaultColumns);
   const auto unknownColumns = FindUnknownColumns(selectedColumns, lm.GetTree(), validCustomColumns,
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


} // namespace TDF
} // namespace Internal
} // namespace ROOT
