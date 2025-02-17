// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RDataSource.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RLogger.hxx"
#include "RtypesCore.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassRef.h"
#include "TError.h" // Info
#include "TInterpreter.h"
#include "TLeaf.h"
#include "TROOT.h" // IsImplicitMTEnabled, GetThreadPoolSize
#include "TTree.h"

#include <stdexcept>
#include <string>
#include <cstring>
#include <typeinfo>

using namespace ROOT::Detail::RDF;
using namespace ROOT::RDF;

ROOT::RLogChannel &ROOT::Detail::RDF::RDFLogChannel()
{
   static RLogChannel c("ROOT.RDF");
   return c;
}

namespace {
using TypeInfoRef = std::reference_wrapper<const std::type_info>;
struct TypeInfoRefHash {
   std::size_t operator()(TypeInfoRef id) const { return id.get().hash_code(); }
};

struct TypeInfoRefEqualComp {
   bool operator()(TypeInfoRef left, TypeInfoRef right) const { return left.get() == right.get(); }
};
} // namespace

namespace ROOT {
namespace Internal {
namespace RDF {

/// Return the type_info associated to a name. If the association fails, an
/// exception is thrown.
/// References and pointers are not supported since those cannot be stored in
/// columns.
const std::type_info &TypeName2TypeID(const std::string &name)
{

   const static std::unordered_map<std::string, TypeInfoRef> typeName2TypeIDMap{
      {"char", typeid(char)},
      {"Char_t", typeid(char)},
      {"unsigned char", typeid(unsigned char)},
      {"UChar_t", typeid(unsigned char)},
      {"int", typeid(int)},
      {"Int_t", typeid(int)},
      {"unsigned", typeid(unsigned int)},
      {"unsigned int", typeid(unsigned int)},
      {"UInt_t", typeid(unsigned int)},
      {"short", typeid(short)},
      {"short int", typeid(short)},
      {"Short_t", typeid(short)},
      {"unsigned short", typeid(unsigned short)},
      {"unsigned short int", typeid(unsigned short)},
      {"UShort_t", typeid(unsigned short)},
      {"long", typeid(long)},
      {"long int", typeid(long)},
      {"Long_t", typeid(long)},
      {"unsigned long", typeid(unsigned long)},
      {"unsigned long int", typeid(unsigned long)},
      {"ULong_t", typeid(unsigned long)},
      {"double", typeid(double)},
      {"Double_t", typeid(double)},
      {"float", typeid(float)},
      {"Float_t", typeid(float)},
      {"long long", typeid(long long)},
      {"long long int", typeid(long long)},
      {"Long64_t", typeid(long long)},
      {"unsigned long long", typeid(unsigned long long)},
      {"unsigned long long int", typeid(unsigned long long)},
      {"ULong64_t", typeid(unsigned long long)},
      {"bool", typeid(bool)},
      {"Bool_t", typeid(bool)}};

   if (auto it = typeName2TypeIDMap.find(name); it != typeName2TypeIDMap.end())
      return it->second.get();

   if (auto c = TClass::GetClass(name.c_str())) {
      if (!c->GetTypeInfo()) {
         throw std::runtime_error("Cannot extract type_info of type " + name + ".");
      }
      return *c->GetTypeInfo();
   }

   throw std::runtime_error("Cannot extract type_info of type " + name + ".");
}

/// Returns the name of a type starting from its type_info
/// An empty string is returned in case of failure
/// References and pointers are not supported since those cannot be stored in
/// columns.
/// Note that this function will take a lock and may be a potential source of
/// contention in multithreaded execution.
std::string TypeID2TypeName(const std::type_info &id)
{
   const static std::unordered_map<TypeInfoRef, std::string, TypeInfoRefHash, TypeInfoRefEqualComp> typeID2TypeNameMap{
      {typeid(char), "char"},         {typeid(unsigned char), "unsigned char"},
      {typeid(int), "int"},           {typeid(unsigned int), "unsigned int"},
      {typeid(short), "short"},       {typeid(unsigned short), "unsigned short"},
      {typeid(long), "long"},         {typeid(unsigned long), "unsigned long"},
      {typeid(double), "double"},     {typeid(float), "float"},
      {typeid(Long64_t), "Long64_t"}, {typeid(ULong64_t), "ULong64_t"},
      {typeid(bool), "bool"}};

   if (auto it = typeID2TypeNameMap.find(id); it != typeID2TypeNameMap.end())
      return it->second;

   if (auto c = TClass::GetClass(id)) {
      return c->GetName();
   }

   return "";
}

std::string ComposeRVecTypeName(const std::string &valueType)
{
   return "ROOT::VecOps::RVec<" + valueType + ">";
}

std::string GetLeafTypeName(TLeaf *leaf, const std::string &colName)
{
   const char *colTypeCStr = leaf->GetTypeName();
   std::string colType = colTypeCStr == nullptr ? "" : colTypeCStr;
   if (colType.empty())
      throw std::runtime_error("Could not deduce type of leaf " + colName);
   if (leaf->GetLeafCount() != nullptr && leaf->GetLenStatic() == 1) {
      // this is a variable-sized array
      colType = ComposeRVecTypeName(colType);
   } else if (leaf->GetLeafCount() == nullptr && leaf->GetLenStatic() > 1) {
      // this is a fixed-sized array (we do not differentiate between variable- and fixed-sized arrays)
      colType = ComposeRVecTypeName(colType);
   } else if (leaf->GetLeafCount() != nullptr && leaf->GetLenStatic() > 1) {
      // we do not know how to deal with this branch
      throw std::runtime_error("TTree leaf " + colName +
                               " has both a leaf count and a static length. This is not supported.");
   }

   return colType;
}

/// Return the typename of object colName stored in t, if any. Return an empty string if colName is not in t.
/// Supported cases:
/// - leaves corresponding to single values, variable- and fixed-length arrays, with following syntax:
///   - "leafname", as long as TTree::GetLeaf resolves it
///   - "b1.b2...leafname", as long as TTree::GetLeaf("b1.b2....", "leafname") resolves it
/// - TBranchElements, as long as TTree::GetBranch resolves their names
std::string GetBranchOrLeafTypeName(TTree &t, const std::string &colName)
{
   // look for TLeaf either with GetLeaf(colName) or with GetLeaf(branchName, leafName) (splitting on last dot)
   auto *leaf = t.GetLeaf(colName.c_str());
   if (!leaf)
      leaf = t.FindLeaf(colName.c_str()); // try harder
   if (!leaf) {
      // try splitting branchname and leafname
      const auto dotPos = colName.find_last_of('.');
      const auto hasDot = dotPos != std::string::npos;
      if (hasDot) {
         const auto branchName = colName.substr(0, dotPos);
         const auto leafName = colName.substr(dotPos + 1);
         leaf = t.GetLeaf(branchName.c_str(), leafName.c_str());
      }
   }
   if (leaf)
      return GetLeafTypeName(leaf, std::string(leaf->GetFullName()));

   // we could not find a leaf named colName, so we look for a branch called like this
   auto branch = t.GetBranch(colName.c_str());
   if (!branch)
      branch = t.FindBranch(colName.c_str()); // try harder
   if (branch) {
      static const TClassRef tbranchelement("TBranchElement");
      if (branch->InheritsFrom(tbranchelement)) {
         auto be = static_cast<TBranchElement *>(branch);
         if (auto currentClass = be->GetCurrentClass())
            return currentClass->GetName();
         else {
            // Here we have a special case for getting right the type of data members
            // of classes sorted in TClonesArrays: ROOT-9674
            auto mother = be->GetMother();
            if (mother && mother->InheritsFrom(tbranchelement) && mother != be) {
               auto beMom = static_cast<TBranchElement *>(mother);
               auto beMomClass = beMom->GetClass();
               if (beMomClass && 0 == std::strcmp("TClonesArray", beMomClass->GetName()))
                  return be->GetTypeName();
            }
            return be->GetClassName();
         }
      } else if (branch->IsA() == TBranch::Class() && branch->GetListOfLeaves()->GetEntriesUnsafe() == 1) {
         // normal branch (not a TBranchElement): if it has only one leaf, we pick the type of the leaf:
         // RDF and TTreeReader allow referring to branch.leaf as just branch if branch has only one leaf
         leaf = static_cast<TLeaf *>(branch->GetListOfLeaves()->UncheckedAt(0));
         return GetLeafTypeName(leaf, std::string(leaf->GetFullName()));
      }
   }

   // we could not find a branch or a leaf called colName
   return std::string();
}

/// Return a string containing the type of the given branch. Works both with real TTree branches and with temporary
/// column created by Define. Throws if type name deduction fails.
/// Note that for fixed- or variable-sized c-style arrays the returned type name will be RVec<T>.
/// vector2RVec specifies whether typename 'std::vector<T>' should be converted to 'RVec<T>' or returned as is
std::string ColumnName2ColumnTypeName(const std::string &colName, TTree *tree, RDataSource *ds, RDefineBase *define,
                                      bool vector2RVec)
{
   std::string colType;

   // must check defines first: we want Redefines to have precedence over everything else
   if (define) {
      colType = define->GetTypeName();
   } else if (ds && ds->HasColumn(colName)) {
      colType = ds->GetTypeName(colName);
   } else if (tree) {
      colType = GetBranchOrLeafTypeName(*tree, colName);
      if (vector2RVec && TClassEdit::IsSTLCont(colType) == ROOT::ESTLType::kSTLvector) {
         std::vector<std::string> split;
         int dummy;
         TClassEdit::GetSplit(colType.c_str(), split, dummy);
         auto &valueType = split[1];
         colType = ComposeRVecTypeName(valueType);
      }
   }

   if (colType.empty())
      throw std::runtime_error("Column \"" + colName +
                               "\" is not in a dataset and is not a custom column been defined.");

   return colType;
}

/// Convert type name (e.g. "Float_t") to ROOT type code (e.g. 'F') -- see TBranch documentation.
/// Return a space ' ' in case no match was found.
char TypeName2ROOTTypeName(const std::string &b)
{
   const static std::unordered_map<std::string, char> typeName2ROOTTypeNameMap{{"char", 'B'},
                                                                               {"Char_t", 'B'},
                                                                               {"unsigned char", 'b'},
                                                                               {"UChar_t", 'b'},
                                                                               {"int", 'I'},
                                                                               {"Int_t", 'I'},
                                                                               {"unsigned", 'i'},
                                                                               {"unsigned int", 'i'},
                                                                               {"UInt_t", 'i'},
                                                                               {"short", 'S'},
                                                                               {"short int", 'S'},
                                                                               {"Short_t", 'S'},
                                                                               {"unsigned short", 's'},
                                                                               {"unsigned short int", 's'},
                                                                               {"UShort_t", 's'},
                                                                               {"long", 'G'},
                                                                               {"long int", 'G'},
                                                                               {"Long_t", 'G'},
                                                                               {"unsigned long", 'g'},
                                                                               {"unsigned long int", 'g'},
                                                                               {"ULong_t", 'g'},
                                                                               {"double", 'D'},
                                                                               {"Double_t", 'D'},
                                                                               {"float", 'F'},
                                                                               {"Float_t", 'F'},
                                                                               {"long long", 'L'},
                                                                               {"long long int", 'L'},
                                                                               {"Long64_t", 'L'},
                                                                               {"unsigned long long", 'l'},
                                                                               {"unsigned long long int", 'l'},
                                                                               {"ULong64_t", 'l'},
                                                                               {"bool", 'O'},
                                                                               {"Bool_t", 'O'}};

   if (auto it = typeName2ROOTTypeNameMap.find(b); it != typeName2ROOTTypeNameMap.end())
      return it->second;

   return ' ';
}

unsigned int GetNSlots()
{
   unsigned int nSlots = 1;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      nSlots = ROOT::GetThreadPoolSize();
#endif // R__USE_IMT
   return nSlots;
}

/// Replace occurrences of '.' with '_' in each string passed as argument.
/// An Info message is printed when this happens. Dots at the end of the string are not replaced.
/// An exception is thrown in case the resulting set of strings would contain duplicates.
std::vector<std::string> ReplaceDotWithUnderscore(const std::vector<std::string> &columnNames)
{
   auto newColNames = columnNames;
   for (auto &col : newColNames) {
      const auto dotPos = col.find('.');
      if (dotPos != std::string::npos && dotPos != col.size() - 1 && dotPos != 0u) {
         auto oldName = col;
         std::replace(col.begin(), col.end(), '.', '_');
         if (std::find(columnNames.begin(), columnNames.end(), col) != columnNames.end())
            throw std::runtime_error("Column " + oldName + " would be written as " + col +
                                     " but this column already exists. Please use Alias to select a new name for " +
                                     oldName);
         Info("Snapshot", "Column %s will be saved as %s", oldName.c_str(), col.c_str());
      }
   }

   return newColNames;
}

void InterpreterDeclare(const std::string &code)
{
   R__LOG_DEBUG(10, RDFLogChannel()) << "Declaring the following code to cling:\n\n" << code << '\n';

   if (!gInterpreter->Declare(code.c_str())) {
      const auto msg =
         "\nRDataFrame: An error occurred during just-in-time compilation. The lines above might indicate the cause of "
         "the crash\n All RDF objects that have not run an event loop yet should be considered in an invalid state.\n";
      throw std::runtime_error(msg);
   }
}

void InterpreterCalc(const std::string &code, const std::string &context)
{
   if (code.empty())
      return;

   R__LOG_DEBUG(10, RDFLogChannel()) << "Jitting and executing the following code:\n\n" << code << '\n';

   TInterpreter::EErrorCode errorCode(TInterpreter::kNoError); // storage for cling errors

   auto callCalc = [&errorCode, &context](const std::string &codeSlice) {
      gInterpreter->Calc(codeSlice.c_str(), &errorCode);
      if (errorCode != TInterpreter::EErrorCode::kNoError) {
         std::string msg = "\nAn error occurred during just-in-time compilation";
         if (!context.empty())
            msg += " in " + context;
         msg +=
            ". The lines above might indicate the cause of the crash\nAll RDF objects that have not run their event "
            "loop yet should be considered in an invalid state.\n";
         throw std::runtime_error(msg);
      }
   };

   // Call Calc every 1000 newlines in order to avoid jitting a very large function body, which is slow:
   // see https://github.com/root-project/root/issues/9312 and https://github.com/root-project/root/issues/7604
   std::size_t substr_start = 0;
   std::size_t substr_end = 0;
   while (substr_end != std::string::npos && substr_start != code.size() - 1) {
      for (std::size_t i = 0u; i < 1000u && substr_end != std::string::npos; ++i) {
         substr_end = code.find('\n', substr_end + 1);
      }
      const std::string subs = code.substr(substr_start, substr_end - substr_start);
      substr_start = substr_end;

      callCalc(subs);
   }
}

bool IsInternalColumn(std::string_view colName)
{
   const auto str = colName.data();
   const auto goodPrefix = colName.size() > 3 &&               // has at least more characters than {r,t}df
                           ('r' == str[0] || 't' == str[0]) && // starts with r or t
                           0 == strncmp("df", str + 1, 2);     // 2nd and 3rd letters are df
   return goodPrefix && '_' == colName.back();                 // also ends with '_'
}

unsigned int GetColumnWidth(const std::vector<std::string>& names, const unsigned int minColumnSpace)
{
   auto columnWidth = 0u;
   for (const auto& name : names) {
      const auto length = name.length();
      if (length > columnWidth)
         columnWidth = length;
   }
   columnWidth = (columnWidth / minColumnSpace + 1) * minColumnSpace;
   return columnWidth;
}

void CheckReaderTypeMatches(const std::type_info &colType, const std::type_info &requestedType,
                            const std::string &colName)
{
   // We want to explicitly support the reading of bools as unsigned char, as
   // this is quite common to circumvent the std::vector<bool> specialization.
   const bool explicitlySupported = (colType == typeid(bool) && requestedType == typeid(unsigned char)) ? true : false;

   // Here we compare names and not typeinfos since they may come from two different contexts: a compiled
   // and a jitted one.
   const auto diffTypes = (0 != std::strcmp(colType.name(), requestedType.name()));
   auto inheritedType = [&]() {
      auto colTClass = TClass::GetClass(colType);
      return colTClass && colTClass->InheritsFrom(TClass::GetClass(requestedType));
   };

   if (!explicitlySupported && diffTypes && !inheritedType()) {
      const auto tName = TypeID2TypeName(requestedType);
      const auto colTypeName = TypeID2TypeName(colType);
      std::string errMsg = "RDataFrame: type mismatch: column \"" + colName + "\" is being used as ";
      if (tName.empty()) {
         errMsg += requestedType.name();
         errMsg += " (extracted from type info)";
      } else {
         errMsg += tName;
      }
      errMsg += " but the Define or Vary node advertises it as ";
      if (colTypeName.empty()) {
         auto &id = colType;
         errMsg += id.name();
         errMsg += " (extracted from type info)";
      } else {
         errMsg += colTypeName;
      }
      throw std::runtime_error(errMsg);
   }
}

bool IsStrInVec(const std::string &str, const std::vector<std::string> &vec)
{
   return std::find(vec.cbegin(), vec.cend(), str) != vec.cend();
}

auto RStringCache::Insert(const std::string &string) -> decltype(fStrings)::const_iterator
{
   {
      std::shared_lock l{fMutex};
      if (auto it = fStrings.find(string); it != fStrings.end())
         return it;
   }

   // TODO: Would be nicer to use a lock upgrade strategy a-la TVirtualRWMutex
   // but that is unfortunately not usable outside the already available ROOT mutexes
   std::unique_lock l{fMutex};
   if (auto it = fStrings.find(string); it != fStrings.end())
      return it;

   return fStrings.insert(string).first;
}
} // end NS RDF
} // end NS Internal
} // end NS ROOT
