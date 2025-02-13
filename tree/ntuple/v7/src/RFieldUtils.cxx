/// \file RFieldUtils.cxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include "RFieldUtils.hxx"

#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleUtil.hxx>

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

// Recursively normalizes a template argument using the regular type name normalizer F as a helper.
template <typename F>
std::string GetNormalizedTemplateArg(const std::string &arg, F fnTypeNormalizer)
{
   R__ASSERT(!arg.empty());

   if (std::isdigit(arg[0]) || arg[0] == '-') {
      // Integer template argument
      return ROOT::Experimental::Internal::GetNormalizedInteger(arg);
   }

   std::string qualifier;
   // Type name template argument; template arguments must keep their CV qualifier
   if (arg.substr(0, 6) == "const " || (arg.length() > 14 && arg.substr(9, 6) == "const "))
      qualifier += "const ";
   if (arg.substr(0, 9) == "volatile " || (arg.length() > 14 && arg.substr(6, 9) == "volatile "))
      qualifier += "volatile ";
   return qualifier + fnTypeNormalizer(arg);
}

std::pair<std::string, std::string> SplitTypePrefixFromTemplateArgs(const std::string &typeName)
{
   auto idxOpen = typeName.find_first_of("<");
   if (idxOpen == std::string::npos)
      return {typeName, ""};

   R__ASSERT(idxOpen > 0);
   R__ASSERT(typeName.back() == '>');
   R__ASSERT((typeName.size() - 1) > idxOpen);

   return {typeName.substr(0, idxOpen), typeName.substr(idxOpen + 1, typeName.size() - idxOpen - 2)};
}

} // namespace

std::string ROOT::Experimental::Internal::GetCanonicalTypePrefix(const std::string &typeName)
{
   std::string canonicalType{TClassEdit::CleanType(typeName.c_str(), /*mode=*/1)};
   if (canonicalType.substr(0, 7) == "struct ") {
      canonicalType.erase(0, 7);
   } else if (canonicalType.substr(0, 5) == "enum ") {
      canonicalType.erase(0, 5);
   } else if (canonicalType.substr(0, 2) == "::") {
      canonicalType.erase(0, 2);
   }

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

   // Map fundamental integer types to stdint integer types (e.g. int --> std::int32_t)
   if (canonicalType == "signed char") {
      canonicalType = RField<signed char>::TypeName();
   } else if (canonicalType == "unsigned char") {
      canonicalType = RField<unsigned char>::TypeName();
   } else if (canonicalType == "short" || canonicalType == "short int" || canonicalType == "signed short" ||
              canonicalType == "signed short int") {
      canonicalType = RField<short int>::TypeName();
   } else if (canonicalType == "unsigned short" || canonicalType == "unsigned short int") {
      canonicalType = RField<unsigned short int>::TypeName();
   } else if (canonicalType == "int" || canonicalType == "signed" || canonicalType == "signed int") {
      canonicalType = RField<int>::TypeName();
   } else if (canonicalType == "unsigned" || canonicalType == "unsigned int") {
      canonicalType = RField<unsigned int>::TypeName();
   } else if (canonicalType == "long" || canonicalType == "long int" || canonicalType == "signed long" ||
              canonicalType == "signed long int") {
      canonicalType = RField<long int>::TypeName();
   } else if (canonicalType == "unsigned long" || canonicalType == "unsigned long int") {
      canonicalType = RField<unsigned long int>::TypeName();
   } else if (canonicalType == "long long" || canonicalType == "long long int" || canonicalType == "signed long long" ||
              canonicalType == "signed long long int") {
      canonicalType = RField<long long int>::TypeName();
   } else if (canonicalType == "unsigned long long" || canonicalType == "unsigned long long int") {
      canonicalType = RField<unsigned long long int>::TypeName();
   }

   return canonicalType;
}

std::string ROOT::Experimental::Internal::GetRenormalizedTypeName(const std::string &metaNormalizedName)
{
   std::string normName{GetCanonicalTypePrefix(metaNormalizedName)};
   // RNTuple resolves Double32_t for the normalized type name but keeps Double32_t for the type alias
   // (also in template parameters)
   if (normName == "Double32_t")
      return "double";

   const auto [typePrefix, argList] = SplitTypePrefixFromTemplateArgs(normName);
   if (argList.empty())
      return typePrefix;

   auto templateArgs = TokenizeTypeList(argList);
   R__ASSERT(!templateArgs.empty());

   normName = typePrefix + "<";
   for (const auto &a : templateArgs) {
      normName += GetNormalizedTemplateArg(a, GetRenormalizedTypeName) + ",";
   }
   normName[normName.size() - 1] = '>';

   return normName;
}

std::string ROOT::Experimental::Internal::GetNormalizedUnresolvedTypeName(const std::string &origName)
{
   const TClassEdit::EModType modType = static_cast<TClassEdit::EModType>(
      TClassEdit::kDropStlDefault | TClassEdit::kDropComparator | TClassEdit::kDropHash);
   std::string normName{origName};
   TClassEdit::TSplitType splitname(normName.c_str(), modType);
   splitname.ShortType(normName, modType);
   normName = GetCanonicalTypePrefix(normName);

   const auto [typePrefix, argList] = SplitTypePrefixFromTemplateArgs(normName);
   if (argList.empty())
      return normName;

   auto templateArgs = TokenizeTypeList(argList);
   R__ASSERT(!templateArgs.empty());

   // Get default-initialized template arguments; we only need to do this for user-defined class types
   auto expandedName = normName;
   if ((expandedName.substr(0, 5) != "std::") && (expandedName.substr(0, 19) != "ROOT::VecOps::RVec<")) {
      auto cl = TClass::GetClass(origName.c_str());
      if (cl)
         expandedName = cl->GetName();
   }
   auto expandedTemplateArgs = TokenizeTypeList(SplitTypePrefixFromTemplateArgs(expandedName).second);
   R__ASSERT(expandedTemplateArgs.size() >= templateArgs.size());

   normName = typePrefix + "<";
   for (const auto &a : templateArgs) {
      normName += GetNormalizedTemplateArg(a, GetNormalizedUnresolvedTypeName) + ",";
   }
   for (std::size_t i = templateArgs.size(); i < expandedTemplateArgs.size(); ++i) {
      normName += GetNormalizedTemplateArg(expandedTemplateArgs[i], GetNormalizedUnresolvedTypeName) + ",";
   }
   normName[normName.size() - 1] = '>';

   return normName;
}

std::string ROOT::Experimental::Internal::GetNormalizedInteger(long long val)
{
   return std::to_string(val);
}

std::string ROOT::Experimental::Internal::GetNormalizedInteger(unsigned long long val)
{
   if (val > std::numeric_limits<std::int64_t>::max())
      return std::to_string(val) + "u";
   return std::to_string(val);
}

std::string ROOT::Experimental::Internal::GetNormalizedInteger(const std::string &intTemplateArg)
{
   R__ASSERT(!intTemplateArg.empty());
   if (intTemplateArg[0] == '-')
      return GetNormalizedInteger(ParseIntTypeToken(intTemplateArg));
   return GetNormalizedInteger(ParseUIntTypeToken(intTemplateArg));
}

long long ROOT::Experimental::Internal::ParseIntTypeToken(const std::string &intToken)
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

unsigned long long ROOT::Experimental::Internal::ParseUIntTypeToken(const std::string &uintToken)
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

ROOT::Experimental::Internal::ERNTupleSerializationMode
ROOT::Experimental::Internal::GetRNTupleSerializationMode(TClass *cl)
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

std::tuple<std::string, std::vector<std::size_t>>
ROOT::Experimental::Internal::ParseArrayType(const std::string &typeName)
{
   std::vector<std::size_t> sizeVec;

   // Only parse outer array definition, i.e. the right `]` should be at the end of the type name
   std::string prefix{typeName};
   while (prefix.back() == ']') {
      auto posRBrace = prefix.size() - 1;
      auto posLBrace = prefix.find_last_of('[', posRBrace);
      if (posLBrace == std::string_view::npos) {
         throw RException(R__FAIL(std::string("invalid array type: ") + typeName));
      }

      const std::size_t size = ParseUIntTypeToken(prefix.substr(posLBrace + 1, posRBrace - posLBrace - 1));
      if (size == 0) {
         throw RException(R__FAIL(std::string("invalid array size: ") + typeName));
      }

      sizeVec.insert(sizeVec.begin(), size);
      prefix.resize(posLBrace);
   }
   return std::make_tuple(prefix, sizeVec);
}

std::vector<std::string> ROOT::Experimental::Internal::TokenizeTypeList(std::string_view templateType)
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
            typeBegin = typeCursor + 1;
         }
         break;
      }
      typeCursor++;
   }
   result.push_back(std::string(typeBegin, typeCursor - typeBegin));
   return result;
}
