// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClass.h"
#include "TRegexp.h"

#include "ROOT/TDFInterface.hxx"

#include <vector>
#include <string>
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Internal::TDF;
using namespace ROOT::Detail::TDF;

namespace ROOT {
namespace Experimental {
namespace TDF {
// extern templates
template class TInterface<TLoopManager>;
template class TInterface<TFilterBase>;
template class TInterface<TCustomColumnBase>;
}
}

namespace Internal {
namespace TDF {
// Match expression against names of branches passed as parameter
// Return vector of names of the branches used in the expression
std::vector<std::string> GetUsedBranchesNames(const std::string expression, TObjArray *branches,
                                              const std::vector<std::string> &tmpBranches)
{
   // Check what branches and temporary branches are used in the expression
   // To help matching the regex
   std::string paddedExpr = " " + expression + " ";
   int paddedExprLen = paddedExpr.size();
   static const std::string regexBit("[^a-zA-Z0-9_]");
   std::vector<std::string> usedBranches;
   for (auto bro : *branches) {
      auto brName = bro->GetName();
      std::string bNameRegexContent = regexBit + brName + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
         usedBranches.emplace_back(brName);
      }
   }
   for (auto brName : tmpBranches) {
      std::string bNameRegexContent = regexBit + brName + regexBit;
      TRegexp bNameRegex(bNameRegexContent.c_str());
      if (-1 != bNameRegex.Index(paddedExpr.c_str(), &paddedExprLen)) {
         usedBranches.emplace_back(brName.c_str());
      }
   }
   return usedBranches;
}

// Jit a string filter or a string temporary column, call this->Define or this->Filter as needed
// Return pointer to the new functional chain node returned by the call, cast to Long_t
Long_t JitTransformation(void *thisPtr, const std::string &methodName, const std::string &nodeTypeName,
                         const std::string &name, const std::string &expression, TObjArray *branches,
                         const std::vector<std::string> &tmpBranches,
                         const std::map<std::string, TmpBranchBasePtr_t> &tmpBookedBranches, TTree *tree)
{
   auto usedBranches = GetUsedBranchesNames(expression, branches, tmpBranches);
   auto exprNeedsVariables = !usedBranches.empty();

   // Move to the preparation of the jitting
   // We put all of the jitted entities in a namespace called
   // __tdf_filter_N, where N is a monotonically increasing index.
   TInterpreter::EErrorCode interpErrCode;
   std::vector<std::string> usedBranchesTypes;
   std::stringstream ss;
   static unsigned int iNs = 0U;
   ss << "__tdf_" << iNs++;
   const auto nsName = ss.str();
   ss.str("");

   if (exprNeedsVariables) {
      // Declare a namespace and inside it the variables in the expression
      ss << "namespace " << nsName;
      ss << " {\n";
      for (auto brName : usedBranches) {
         // The map is a const reference, so no operator[]
         auto tmpBrIt = tmpBookedBranches.find(brName);
         auto tmpBr = tmpBrIt == tmpBookedBranches.end() ? nullptr : tmpBrIt->second.get();
         auto brTypeName = ColumnName2ColumnTypeName(brName, *tree, tmpBr);
         ss << brTypeName << " " << brName << ";\n";
         usedBranchesTypes.emplace_back(brTypeName);
      }
      ss << "}";
      auto variableDeclarations = ss.str();
      ss.str("");
      // We need ProcessLine to trigger auto{parsing,loading} where needed
      gInterpreter->ProcessLine(variableDeclarations.c_str(), &interpErrCode);
      if (TInterpreter::EErrorCode::kNoError != interpErrCode) {
         std::string msg = "Cannot declare these variables ";
         msg += " ";
         msg += variableDeclarations;
         if (TInterpreter::EErrorCode::kNoError != interpErrCode) {
            msg += "\nInterpreter error code is " + std::to_string(interpErrCode) + ".";
         }
         throw std::runtime_error(msg);
      }
   }

   // Declare within the same namespace, the expression to make sure it
   // is proper C++
   ss << "namespace " << nsName << "{ auto res = " << expression << ";}\n";
   // Headers must have been parsed and libraries loaded: we can use Declare
   if (!gInterpreter->Declare(ss.str().c_str())) {
      std::string msg = "Cannot interpret this expression: ";
      msg += " ";
      msg += ss.str();
      throw std::runtime_error(msg);
   }

   // Now we build the lambda and we invoke the method with it in the jitted world
   ss.str("");
   ss << "[](";
   for (unsigned int i = 0; i < usedBranchesTypes.size(); ++i) {
      // We pass by reference to avoid expensive copies
      ss << usedBranchesTypes[i] << "& " << usedBranches[i] << ", ";
   }
   if (!usedBranchesTypes.empty()) ss.seekp(-2, ss.cur);
   ss << "){ return " << expression << ";}";
   auto filterLambda = ss.str();

   // Here we have two cases: filter and column
   ss.str("");
   ss << "((" << nodeTypeName << "*)" << thisPtr << ")->" << methodName << "(";
   if (methodName == "Define") {
      ss << "\"" << name << "\", ";
   }
   ss << filterLambda << ", {";
   for (auto brName : usedBranches) {
      ss << "\"" << brName << "\", ";
   }
   if (exprNeedsVariables) ss.seekp(-2, ss.cur); // remove the last ",
   ss << "}";

   if (methodName == "Filter") {
      ss << ", \"" << name << "\"";
   }

   ss << ");";

   auto retVal = gInterpreter->ProcessLine(ss.str().c_str(), &interpErrCode);
   if (TInterpreter::EErrorCode::kNoError != interpErrCode || !retVal) {
      std::string msg = "Cannot interpret the invocation to " + methodName + ": ";
      msg += " ";
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
void JitBuildAndBook(const ColumnNames_t &bl, const std::string &nodeTypename, void *thisPtr, const std::type_info &art,
                     const std::type_info &at, const void *r, TTree &tree, unsigned int nSlots,
                     const std::map<std::string, TmpBranchBasePtr_t> &tmpBranches)
{
   gInterpreter->ProcessLine("#include \"ROOT/TDataFrame.hxx\"");
   auto nBranches = bl.size();

   // retrieve pointers to temporary columns (null if the column is not temporary)
   std::vector<TCustomColumnBase *> tmpBranchPtrs(nBranches, nullptr);
   for (auto i = 0u; i < nBranches; ++i) {
      auto tmpBranchIt = tmpBranches.find(bl[i]);
      if (tmpBranchIt != tmpBranches.end()) tmpBranchPtrs[i] = tmpBranchIt->second.get();
   }

   // retrieve branch type names as strings
   std::vector<std::string> branchTypeNames(nBranches);
   for (auto i = 0u; i < nBranches; ++i) {
      const auto branchTypeName = ColumnName2ColumnTypeName(bl[i], tree, tmpBranchPtrs[i]);
      if (branchTypeName.empty()) {
         std::string exceptionText = "The type of column ";
         exceptionText += bl[i];
         exceptionText += " could not be guessed. Please specify one.";
         throw std::runtime_error(exceptionText.c_str());
      }
      branchTypeNames[i] = branchTypeName;
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
   // ROOT::Internal::TDF::CallBuildAndBook<nodeType, actionType, branchType1, branchType2...>(
   //    reinterpret_cast<nodeType*>(thisPtr), *reinterpret_cast<ROOT::ColumnNames_t*>(&bl),
   //    *reinterpret_cast<actionResultType*>(r), reinterpret_cast<ActionType*>(nullptr))
   std::stringstream createAction_str;
   createAction_str << "ROOT::Internal::TDF::CallBuildAndBook<" << nodeTypename << ", " << actionTypeName;
   for (auto &branchTypeName : branchTypeNames) createAction_str << ", " << branchTypeName;
   createAction_str << ">("
                    << "reinterpret_cast<" << nodeTypename << "*>(" << thisPtr << "), "
                    << "*reinterpret_cast<ROOT::Detail::TDF::ColumnNames_t*>(" << &bl << "), " << nSlots
                    << ", *reinterpret_cast<" << actionResultTypeName << "*>(" << r << "));";
   auto error = TInterpreter::EErrorCode::kNoError;
   gInterpreter->ProcessLine(createAction_str.str().c_str(), &error);
   if (error) {
      std::string exceptionText = "An error occurred while jitting this action:\n";
      exceptionText += createAction_str.str();
      throw std::runtime_error(exceptionText.c_str());
   }
}
} // end ns TDF
} // end ns Internal
} // end ns ROOT
