// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRegexp.h"

#include "ROOT/TDataFrameInterface.hxx"

#include <vector>
#include <string>

namespace ROOT {

namespace Internal {
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

Long_t InterpretCall(void *thisPtr, const std::string &methodName, const std::string &nodeTypeName,
                     const std::string &name, const std::string &expression, TObjArray *branches,
                     const std::vector<std::string> &tmpBranches,
                     const std::map<std::string, TmpBranchBasePtr_t> &tmpBookedBranches, TTree *tree)
{
   auto usedBranches = ROOT::Internal::GetUsedBranchesNames(expression, branches, tmpBranches);
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
         auto brTypeName = ROOT::Internal::ColumnName2ColumnTypeName(brName, *tree, tmpBr);
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
   if ("AddColumn" == methodName) {
      ss << "\"" << name << "\", ";
   }
   ss << filterLambda << ", {";
   for (auto brName : usedBranches) {
      ss << "\"" << brName << "\", ";
   }
   if (exprNeedsVariables) ss.seekp(-2, ss.cur); // remove the last ",
   ss << "}";

   if ("Filter" == methodName) {
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
}

namespace Experimental {

// extern templates
template class TDataFrameInterface<ROOT::Detail::TDataFrameImpl>;
template class TDataFrameInterface<ROOT::Detail::TDataFrameFilterBase>;
template class TDataFrameInterface<ROOT::Detail::TDataFrameBranchBase>;

} // namespace Experimental
} // namespace ROOT
