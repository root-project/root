/// \file RFieldUtils.cxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19

#include <ROOT/RFieldUtils.hxx>

#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>

#include <TClass.h>
#include <TClassEdit.h>
#include <TDictAttributeMap.h>

#include <algorithm>
#include <charconv>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::string GetRenormalizedDemangledTypeName(const std::string &demangledName, bool renormalizeStdString);

const std::unordered_map<std::string_view, std::string_view> typeTranslationMap{
   {"Bool_t", "bool"},
   {"Float_t", "float"},
   {"Double_t", "double"},
   {"string", "std::string"},

   {"byte", "std::byte"},
   {"Char_t", "char"},
   {"int8_t", "std::int8_t"},
   {"UChar_t", "unsigned char"},
   {"uint8_t", "std::uint8_t"},

   {"Short_t", "short"},
   {"int16_t", "std::int16_t"},
   {"UShort_t", "unsigned short"},
   {"uint16_t", "std::uint16_t"},

   {"Int_t", "int"},
   {"int32_t", "std::int32_t"},
   {"UInt_t", "unsigned int"},
   {"unsigned", "unsigned int"},
   {"uint32_t", "std::uint32_t"},

   // Long_t and ULong_t follow the platform's size of long and unsigned long: They are 64 bit on 64-bit Linux and
   // macOS, but 32 bit on 32-bit platforms and Windows (regardless of pointer size).
   {"Long_t", "long"},
   {"ULong_t", "unsigned long"},

   {"Long64_t", "long long"},
   {"int64_t", "std::int64_t"},
   {"ULong64_t", "unsigned long long"},
   {"uint64_t", "std::uint64_t"}};

// Natively supported types drop the default template arguments and the CV qualifiers in template arguments.
bool IsUserClass(const std::string &typeName)
{
   return typeName.rfind("std::", 0) != 0 && typeName.rfind("ROOT::VecOps::RVec<", 0) != 0;
}

/// Parse a type name of the form `T[n][m]...` and return the base type `T` and a vector that contains,
/// in order, the declared size for each dimension, e.g. for `unsigned char[1][2][3]` it returns the tuple
/// `{"unsigned char", {1, 2, 3}}`. Extra whitespace in `typeName` should be removed before calling this function.
///
/// If `typeName` is not an array type, it returns a tuple `{T, {}}`. On error, it returns a default-constructed tuple.
std::tuple<std::string, std::vector<std::size_t>> ParseArrayType(const std::string &typeName)
{
   std::vector<std::size_t> sizeVec;

   // Only parse outer array definition, i.e. the right `]` should be at the end of the type name
   std::string prefix{typeName};
   while (prefix.back() == ']') {
      auto posRBrace = prefix.size() - 1;
      auto posLBrace = prefix.rfind('[', posRBrace);
      if (posLBrace == std::string_view::npos) {
         throw ROOT::RException(R__FAIL(std::string("invalid array type: ") + typeName));
      }
      if (posRBrace - posLBrace <= 1) {
         throw ROOT::RException(R__FAIL(std::string("invalid array type: ") + typeName));
      }

      const std::size_t size =
         ROOT::Internal::ParseUIntTypeToken(prefix.substr(posLBrace + 1, posRBrace - posLBrace - 1));
      if (size == 0) {
         throw ROOT::RException(R__FAIL(std::string("invalid array size: ") + typeName));
      }

      sizeVec.insert(sizeVec.begin(), size);
      prefix.resize(posLBrace);
   }
   return std::make_tuple(prefix, sizeVec);
}

/// Assembles a (nested) std::array<> based type based on the dimensions retrieved from ParseArrayType(). Returns
/// baseType if there are no dimensions.
std::string GetStandardArrayType(const std::string &baseType, const std::vector<std::size_t> &dimensions)
{
   std::string typeName = baseType;
   for (auto i = dimensions.rbegin(), iEnd = dimensions.rend(); i != iEnd; ++i) {
      typeName = "std::array<" + typeName + "," + std::to_string(*i) + ">";
   }
   return typeName;
}

// Recursively normalizes a template argument using the regular type name normalizer F as a helper.
template <typename F>
std::string GetNormalizedTemplateArg(const std::string &arg, bool keepQualifier, F fnTypeNormalizer)
{
   R__ASSERT(!arg.empty());

   if (std::isdigit(arg[0]) || arg[0] == '-') {
      // Integer template argument
      return ROOT::Internal::GetNormalizedInteger(arg);
   }

   if (!keepQualifier)
      return fnTypeNormalizer(arg);

   std::string qualifier;
   // Type name template argument; template arguments must keep their CV qualifier. We assume that fnTypeNormalizer
   // strips the qualifier.
   // Demangled names may have the CV qualifiers suffixed and not prefixed (but const always before volatile).
   // Note that in the latter case, we may have the CV qualifiers before array brackets, e.g. `int const[2]`.
   const auto [base, _] = ParseArrayType(arg);
   if (base.rfind("const ", 0) == 0 || base.rfind("volatile const ", 0) == 0 ||
       base.find(" const", base.length() - 6) != std::string::npos ||
       base.find(" const volatile", base.length() - 15) != std::string::npos) {
      qualifier += "const ";
   }
   if (base.rfind("volatile ", 0) == 0 || base.rfind("const volatile ", 0) == 0 ||
       base.find(" volatile", base.length() - 9) != std::string::npos) {
      qualifier += "volatile ";
   }
   return qualifier + fnTypeNormalizer(arg);
}

using AnglePos = std::pair<std::string::size_type, std::string::size_type>;
std::vector<AnglePos> FindTemplateAngleBrackets(const std::string &typeName)
{
   std::vector<AnglePos> result;
   std::string::size_type currentPos = 0;
   while (currentPos < typeName.size()) {
      const auto posOpen = typeName.find('<', currentPos);
      if (posOpen == std::string::npos) {
         // If there are no more templates, the function is done.
         break;
      }

      auto posClose = posOpen + 1;
      int level = 1;
      while (posClose < typeName.size()) {
         const auto c = typeName[posClose];
         if (c == '<') {
            level++;
         } else if (c == '>') {
            if (level == 1) {
               break;
            }
            level--;
         }
         posClose++;
      }
      // We should have found a closing angle bracket at the right level.
      R__ASSERT(posClose < typeName.size());
      result.emplace_back(posOpen, posClose);

      // If we are not at the end yet, the following two characeters should be :: for nested types.
      if (posClose < typeName.size() - 1) {
         R__ASSERT(typeName.substr(posClose + 1, 2) == "::");
      }
      currentPos = posClose + 1;
   }

   return result;
}

// TClassEdit::CleanType and the name demangling insert blanks between closing angle brackets,
// as they were required before C++11. Name demangling introduces a blank before array dimensions,
// which should also be removed.
void RemoveSpaceBefore(std::string &typeName, char beforeChar)
{
   auto dst = typeName.begin();
   auto end = typeName.end();
   for (auto src = dst; src != end; ++src) {
      if (*src == ' ') {
         auto next = src + 1;
         if (next != end && *next == beforeChar) {
            // Skip this space before a closing angle bracket.
            continue;
         }
      }
      *(dst++) = *src;
   }
   typeName.erase(dst, end);
}

// The demangled name adds spaces after commas
void RemoveSpaceAfter(std::string &typeName, char afterChar)
{
   auto dst = typeName.begin();
   auto end = typeName.end();
   for (auto src = dst; src != end; ++src) {
      *(dst++) = *src;
      if (*src == afterChar) {
         auto next = src + 1;
         if (next != end && *next == ' ') {
            // Skip this space before a closing angle bracket.
            ++src;
         }
      }
   }
   typeName.erase(dst, end);
}

// We normalize typenames to omit any `class`, `struct`, `enum` prefix
void RemoveLeadingKeyword(std::string &typeName)
{
   if (typeName.rfind("class ", 0) == 0) {
      typeName.erase(0, 6);
   } else if (typeName.rfind("struct ", 0) == 0) {
      typeName.erase(0, 7);
   } else if (typeName.rfind("enum ", 0) == 0) {
      typeName.erase(0, 5);
   }
}

// Needed for template arguments in demangled names
void RemoveCVQualifiers(std::string &typeName)
{
   if (typeName.rfind("const ", 0) == 0)
      typeName.erase(0, 6);
   if (typeName.rfind("volatile ", 0) == 0)
      typeName.erase(0, 9);
   if (typeName.find(" volatile", typeName.length() - 9) != std::string::npos)
      typeName.erase(typeName.length() - 9);
   if (typeName.find(" const", typeName.length() - 6) != std::string::npos)
      typeName.erase(typeName.length() - 6);
}

// Map fundamental integer types to stdint integer types (e.g. int --> std::int32_t)
void MapIntegerType(std::string &typeName)
{
   if (typeName == "signed char") {
      typeName = ROOT::RField<signed char>::TypeName();
   } else if (typeName == "unsigned char") {
      typeName = ROOT::RField<unsigned char>::TypeName();
   } else if (typeName == "short" || typeName == "short int" || typeName == "signed short" ||
              typeName == "signed short int") {
      typeName = ROOT::RField<short int>::TypeName();
   } else if (typeName == "unsigned short" || typeName == "unsigned short int") {
      typeName = ROOT::RField<unsigned short int>::TypeName();
   } else if (typeName == "int" || typeName == "signed" || typeName == "signed int") {
      typeName = ROOT::RField<int>::TypeName();
   } else if (typeName == "unsigned" || typeName == "unsigned int") {
      typeName = ROOT::RField<unsigned int>::TypeName();
   } else if (typeName == "long" || typeName == "long int" || typeName == "signed long" ||
              typeName == "signed long int") {
      typeName = ROOT::RField<long int>::TypeName();
   } else if (typeName == "unsigned long" || typeName == "unsigned long int") {
      typeName = ROOT::RField<unsigned long int>::TypeName();
   } else if (typeName == "long long" || typeName == "long long int" || typeName == "signed long long" ||
              typeName == "signed long long int") {
      typeName = ROOT::RField<long long int>::TypeName();
   } else if (typeName == "unsigned long long" || typeName == "unsigned long long int") {
      typeName = ROOT::RField<unsigned long long int>::TypeName();
   } else {
      // The following two types are 64-bit integers on Windows that we can encounter during renormalization of
      // demangled std::type_info names.
      if (typeName == "__int64") {
         typeName = "std::int64_t";
      } else if (typeName == "unsigned __int64") {
         typeName = "std::uint64_t";
      }
   }
}

// Note that ROOT Meta already defines GetDemangledTypeName(), which does both demangling and normalizing.
std::string GetRawDemangledTypeName(const std::type_info &ti)
{
   int e;
   char *str = TClassEdit::DemangleName(ti.name(), e);
   R__ASSERT(str && e == 0);
   std::string result{str};
   free(str);

   return result;
}

// Reverse std::string --> std::basic_string<char> from the demangling
void RenormalizeStdString(std::string &normalizedTypeName)
{
   static const std::string gStringName =
      GetRenormalizedDemangledTypeName(GetRawDemangledTypeName(typeid(std::string)), false /* renormalizeStdString */);

   // For real nested types of std::string (not typedefs like std::string::size_type), we would need to also check
   // something like (normalizedTypeName + "::" == gStringName + "::") and replace only the prefix. However, since
   // such a nested type is not standardized, it currently does not seem necessary to add the logic.
   if (normalizedTypeName == gStringName) {
      normalizedTypeName = "std::string";
   }
}

// Reverse "internal" namespace prefix found in demangled names, such as std::vector<T> --> std::__1::vector<T>
void RenormalizeStdlibType(std::string &normalizedTypeName)
{
   static std::vector<std::pair<std::string, std::string>> gDistortedStdlibNames = []() {
      // clang-format off
      // Listed in order of appearance in the BinaryFormatSpecification.md
      static const std::vector<std::pair<const std::type_info &, std::string>> gCandidates =
         {{typeid(std::vector<char>),                    "std::vector<"},
          {typeid(std::array<char, 1>),                  "std::array<"},
          {typeid(std::variant<char>),                   "std::variant<"},
          {typeid(std::pair<char, char>),                "std::pair<"},
          {typeid(std::tuple<char>),                     "std::tuple<"},
          {typeid(std::bitset<1>),                       "std::bitset<"},
          {typeid(std::unique_ptr<char>),                "std::unique_ptr<"},
          {typeid(std::optional<char>),                  "std::optional<"},
          {typeid(std::set<char>),                       "std::set<"},
          {typeid(std::unordered_set<char>),             "std::unordered_set<"},
          {typeid(std::multiset<char>),                  "std::multiset<"},
          {typeid(std::unordered_multiset<char>),        "std::unordered_multiset<"},
          {typeid(std::map<char, char>),                 "std::map<"},
          {typeid(std::unordered_map<char, char>),       "std::unordered_map<"},
          {typeid(std::multimap<char, char>),            "std::multimap<"},
          {typeid(std::unordered_multimap<char, char>),  "std::unordered_multimap<"},
          {typeid(std::atomic<char>),                    "std::atomic<"}};
      // clang-format on

      std::vector<std::pair<std::string, std::string>> result;
      for (const auto &[ti, prefix] : gCandidates) {
         const auto dm = GetRawDemangledTypeName(ti);
         if (dm.rfind(prefix, 0) == std::string::npos)
            result.push_back(std::make_pair(dm.substr(0, dm.find('<') + 1), prefix));
      }

      return result;
   }();

   for (const auto &[seenPrefix, canonicalPrefix] : gDistortedStdlibNames) {
      if (normalizedTypeName.rfind(seenPrefix, 0) == 0) {
         normalizedTypeName = canonicalPrefix + normalizedTypeName.substr(seenPrefix.length());
         break;
      }
   }
}

template <typename F>
void NormalizeTemplateArguments(std::string &templatedTypeName, int maxTemplateArgs, F fnTypeNormalizer)
{
   const auto angleBrackets = FindTemplateAngleBrackets(templatedTypeName);
   R__ASSERT(!angleBrackets.empty());

   std::string normName;
   std::string::size_type currentPos = 0;
   for (std::size_t i = 0; i < angleBrackets.size(); i++) {
      const auto [posOpen, posClose] = angleBrackets[i];
      // Append the type prefix until the open angle bracket.
      normName += templatedTypeName.substr(currentPos, posOpen + 1 - currentPos);

      const auto argList = templatedTypeName.substr(posOpen + 1, posClose - posOpen - 1);
      const auto templateArgs = ROOT::Internal::TokenizeTypeList(argList, maxTemplateArgs);
      R__ASSERT(!templateArgs.empty());

      const bool isUserClass = IsUserClass(templatedTypeName);
      for (const auto &a : templateArgs) {
         normName += GetNormalizedTemplateArg(a, isUserClass, fnTypeNormalizer) + ",";
      }

      normName[normName.size() - 1] = '>';
      currentPos = posClose + 1;
   }

   // Append the rest of the type from the last closing angle bracket.
   const auto lastClosePos = angleBrackets.back().second;
   normName += templatedTypeName.substr(lastClosePos + 1);

   templatedTypeName = normName;
}

// Given a type name normalized by ROOT Meta, return the type name normalized according to the RNTuple rules.
std::string GetRenormalizedMetaTypeName(const std::string &metaNormalizedName)
{
   const auto canonicalTypePrefix = ROOT::Internal::GetCanonicalTypePrefix(metaNormalizedName);
   // RNTuple resolves Double32_t for the normalized type name but keeps Double32_t for the type alias
   // (also in template parameters)
   if (canonicalTypePrefix == "Double32_t")
      return "double";

   if (canonicalTypePrefix.find('<') == std::string::npos) {
      // If there are no templates, the function is done.
      return canonicalTypePrefix;
   }

   std::string normName{canonicalTypePrefix};
   NormalizeTemplateArguments(normName, 0 /* maxTemplateArgs */, GetRenormalizedMetaTypeName);

   return normName;
}

// Given a demangled name ("normalized by the compiler"), return the type name normalized according to the
// RNTuple rules.
std::string GetRenormalizedDemangledTypeName(const std::string &demangledName, bool renormalizeStdString)
{
   std::string tn{demangledName};
   RemoveSpaceBefore(tn, '[');
   auto [canonicalTypePrefix, dimensions] = ParseArrayType(tn);
   RemoveCVQualifiers(canonicalTypePrefix);
   RemoveLeadingKeyword(canonicalTypePrefix);
   MapIntegerType(canonicalTypePrefix);

   if (canonicalTypePrefix.find('<') == std::string::npos) {
      // If there are no templates, the function is done.
      return GetStandardArrayType(canonicalTypePrefix, dimensions);
   }
   RemoveSpaceBefore(canonicalTypePrefix, '>');
   RemoveSpaceAfter(canonicalTypePrefix, ',');
   RemoveSpaceBefore(canonicalTypePrefix, ','); // MSVC fancies spaces before commas in the demangled name
   RenormalizeStdlibType(canonicalTypePrefix);

   // Remove optional stdlib template arguments
   int maxTemplateArgs = 0;
   if (canonicalTypePrefix.rfind("std::vector<", 0) == 0 || canonicalTypePrefix.rfind("std::set<", 0) == 0 ||
       canonicalTypePrefix.rfind("std::unordered_set<", 0) == 0 ||
       canonicalTypePrefix.rfind("std::multiset<", 0) == 0 ||
       canonicalTypePrefix.rfind("std::unordered_multiset<", 0) == 0 ||
       canonicalTypePrefix.rfind("std::unique_ptr<", 0) == 0) {
      maxTemplateArgs = 1;
   } else if (canonicalTypePrefix.rfind("std::map<", 0) == 0 ||
              canonicalTypePrefix.rfind("std::unordered_map<", 0) == 0 ||
              canonicalTypePrefix.rfind("std::multimap<", 0) == 0 ||
              canonicalTypePrefix.rfind("std::unordered_multimap<", 0) == 0) {
      maxTemplateArgs = 2;
   }

   std::string normName{canonicalTypePrefix};
   NormalizeTemplateArguments(normName, maxTemplateArgs, [renormalizeStdString](const std::string &n) {
      return GetRenormalizedDemangledTypeName(n, renormalizeStdString);
   });
   // In RenormalizeStdString(), we normalize the demangled type name of `std::string`,
   // so we need to prevent an endless recursion.
   if (renormalizeStdString) {
      RenormalizeStdString(normName);
   }

   return GetStandardArrayType(normName, dimensions);
}

} // namespace

std::string ROOT::Internal::GetCanonicalTypePrefix(const std::string &typeName)
{
   // Remove outer cv qualifiers and extra white spaces
   const std::string cleanedType = TClassEdit::CleanType(typeName.c_str(), /*mode=*/1);

   // Can happen when called from RFieldBase::Create() and is caught there
   if (cleanedType.empty())
      return "";

   auto [canonicalType, dimensions] = ParseArrayType(cleanedType);

   RemoveLeadingKeyword(canonicalType);
   if (canonicalType.substr(0, 2) == "::") {
      canonicalType.erase(0, 2);
   }

   RemoveSpaceBefore(canonicalType, '>');

   if (canonicalType.substr(0, 6) == "array<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 7) == "atomic<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 7) == "bitset<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 4) == "map<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 9) == "multimap<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 9) == "multiset<") {
      canonicalType = "std::" + canonicalType;
   }
   if (canonicalType.substr(0, 5) == "pair<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 4) == "set<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 6) == "tuple<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 11) == "unique_ptr<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 14) == "unordered_map<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 19) == "unordered_multimap<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 19) == "unordered_multiset<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 14) == "unordered_set<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 8) == "variant<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 7) == "vector<") {
      canonicalType = "std::" + canonicalType;
   } else if (canonicalType.substr(0, 11) == "ROOT::RVec<") {
      canonicalType = "ROOT::VecOps::RVec<" + canonicalType.substr(11);
   }

   if (auto it = typeTranslationMap.find(canonicalType); it != typeTranslationMap.end()) {
      canonicalType = it->second;
   }

   MapIntegerType(canonicalType);

   return GetStandardArrayType(canonicalType, dimensions);
}

std::string ROOT::Internal::GetRenormalizedTypeName(const std::type_info &ti)
{
   return GetRenormalizedDemangledTypeName(GetRawDemangledTypeName(ti), true /* renormalizeStdString */);
}

std::string ROOT::Internal::GetRenormalizedTypeName(const std::string &metaNormalizedName)
{
   return GetRenormalizedMetaTypeName(metaNormalizedName);
}

std::string ROOT::Internal::GetNormalizedUnresolvedTypeName(const std::string &origName)
{
   const TClassEdit::EModType modType = static_cast<TClassEdit::EModType>(
      TClassEdit::kDropStlDefault | TClassEdit::kDropComparator | TClassEdit::kDropHash);
   TClassEdit::TSplitType splitname(origName.c_str(), modType);
   std::string shortType;
   splitname.ShortType(shortType, modType);
   const auto canonicalTypePrefix = GetCanonicalTypePrefix(shortType);

   if (canonicalTypePrefix.find('<') == std::string::npos) {
      // If there are no templates, the function is done.
      return canonicalTypePrefix;
   }

   const auto angleBrackets = FindTemplateAngleBrackets(canonicalTypePrefix);
   R__ASSERT(!angleBrackets.empty());

   // For user-defined class types, we will need to get the default-initialized template arguments.
   const bool isUserClass = IsUserClass(canonicalTypePrefix);

   std::string normName;
   std::string::size_type currentPos = 0;
   for (std::size_t i = 0; i < angleBrackets.size(); i++) {
      const auto [posOpen, posClose] = angleBrackets[i];
      // Append the type prefix until the open angle bracket.
      normName += canonicalTypePrefix.substr(currentPos, posOpen + 1 - currentPos);

      const auto argList = canonicalTypePrefix.substr(posOpen + 1, posClose - posOpen - 1);
      const auto templateArgs = TokenizeTypeList(argList);
      R__ASSERT(!templateArgs.empty());

      for (const auto &a : templateArgs) {
         normName += GetNormalizedTemplateArg(a, isUserClass, GetNormalizedUnresolvedTypeName) + ",";
      }

      // For user-defined classes, append default-initialized template arguments.
      if (isUserClass) {
         const auto cl = TClass::GetClass(canonicalTypePrefix.substr(0, posClose + 1).c_str());
         if (cl) {
            const std::string expandedName = cl->GetName();
            const auto expandedAngleBrackets = FindTemplateAngleBrackets(expandedName);
            // We can have fewer pairs than angleBrackets, for example in case of type aliases.
            R__ASSERT(!expandedAngleBrackets.empty());

            const auto [expandedPosOpen, expandedPosClose] = expandedAngleBrackets.back();
            const auto expandedArgList =
               expandedName.substr(expandedPosOpen + 1, expandedPosClose - expandedPosOpen - 1);
            const auto expandedTemplateArgs = TokenizeTypeList(expandedArgList);
            // Note that we may be in a sitation where expandedTemplateArgs.size() is _smaller_ than
            // templateArgs.size(), which is when the input type name has the optional template arguments explicitly
            // spelled out but ROOT Meta is told to ignore some template arguments.

            for (std::size_t j = templateArgs.size(); j < expandedTemplateArgs.size(); ++j) {
               normName +=
                  GetNormalizedTemplateArg(expandedTemplateArgs[j], isUserClass, GetNormalizedUnresolvedTypeName) + ",";
            }
         }
      }

      normName[normName.size() - 1] = '>';
      currentPos = posClose + 1;
   }

   // Append the rest of the type from the last closing angle bracket.
   const auto lastClosePos = angleBrackets.back().second;
   normName += canonicalTypePrefix.substr(lastClosePos + 1);

   return normName;
}

std::string ROOT::Internal::GetNormalizedInteger(long long val)
{
   return std::to_string(val);
}

std::string ROOT::Internal::GetNormalizedInteger(unsigned long long val)
{
   if (val > std::numeric_limits<std::int64_t>::max())
      return std::to_string(val) + "u";
   return std::to_string(val);
}

std::string ROOT::Internal::GetNormalizedInteger(const std::string &intTemplateArg)
{
   R__ASSERT(!intTemplateArg.empty());
   if (intTemplateArg[0] == '-')
      return GetNormalizedInteger(ParseIntTypeToken(intTemplateArg));
   return GetNormalizedInteger(ParseUIntTypeToken(intTemplateArg));
}

long long ROOT::Internal::ParseIntTypeToken(const std::string &intToken)
{
   std::size_t nChars = 0;
   long long res = std::stoll(intToken, &nChars);
   if (nChars == intToken.size())
      return res;

   assert(nChars < intToken.size());
   if (nChars == 0) {
      throw RException(R__FAIL("invalid integer type token: " + intToken));
   }

   auto suffix = intToken.substr(nChars);
   std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);
   if (suffix == "L" || suffix == "LL")
      return res;
   if (res >= 0 && (suffix == "U" || suffix == "UL" || suffix == "ULL"))
      return res;

   throw RException(R__FAIL("invalid integer type token: " + intToken));
}

unsigned long long ROOT::Internal::ParseUIntTypeToken(const std::string &uintToken)
{
   std::size_t nChars = 0;
   unsigned long long res = std::stoull(uintToken, &nChars);
   if (nChars == uintToken.size())
      return res;

   assert(nChars < uintToken.size());
   if (nChars == 0) {
      throw RException(R__FAIL("invalid integer type token: " + uintToken));
   }

   auto suffix = uintToken.substr(nChars);
   std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);
   if (suffix == "U" || suffix == "L" || suffix == "LL" || suffix == "UL" || suffix == "ULL")
      return res;

   throw RException(R__FAIL("invalid integer type token: " + uintToken));
}

ROOT::Internal::ERNTupleSerializationMode ROOT::Internal::GetRNTupleSerializationMode(TClass *cl)
{
   auto am = cl->GetAttributeMap();
   if (!am || !am->HasKey("rntuple.streamerMode"))
      return ERNTupleSerializationMode::kUnset;

   std::string value = am->GetPropertyAsString("rntuple.streamerMode");
   std::transform(value.begin(), value.end(), value.begin(), ::toupper);
   if (value == "TRUE") {
      return ERNTupleSerializationMode::kForceStreamerMode;
   } else if (value == "FALSE") {
      return ERNTupleSerializationMode::kForceNativeMode;
   } else {
      R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "invalid setting for 'rntuple.streamerMode' class attribute: "
                                                  << am->GetPropertyAsString("rntuple.streamerMode");
      return ERNTupleSerializationMode::kUnset;
   }
}

std::vector<std::string> ROOT::Internal::TokenizeTypeList(std::string_view templateType, std::size_t maxArgs)
{
   std::vector<std::string> result;
   if (templateType.empty())
      return result;

   const char *eol = templateType.data() + templateType.length();
   const char *typeBegin = templateType.data();
   const char *typeCursor = templateType.data();
   unsigned int nestingLevel = 0;
   while (typeCursor != eol) {
      switch (*typeCursor) {
      case '<': ++nestingLevel; break;
      case '>': --nestingLevel; break;
      case ',':
         if (nestingLevel == 0) {
            result.push_back(std::string(typeBegin, typeCursor - typeBegin));
            if (maxArgs && result.size() == maxArgs)
               return result;
            typeBegin = typeCursor + 1;
         }
         break;
      }
      typeCursor++;
   }
   result.push_back(std::string(typeBegin, typeCursor - typeBegin));
   return result;
}

bool ROOT::Internal::IsMatchingFieldType(std::string_view actualTypeName, std::string_view expectedTypeName,
                                         const std::type_info &ti)
{
   // Fast path: the caller provided the expected type name (from RField<T>::TypeName())
   if (actualTypeName == expectedTypeName)
      return true;

   // The type name may be equal to the alternative, short type name issued by Meta. This is a rare case used, e.g.,
   // by the ATLAS DataVector class to hide a default template parameter from the on-disk type name.
   // Thus, we check again using first ROOT Meta normalization followed by RNTuple re-normalization.
   return (actualTypeName == ROOT::Internal::GetRenormalizedTypeName(ROOT::Internal::GetDemangledTypeName(ti)));
}
