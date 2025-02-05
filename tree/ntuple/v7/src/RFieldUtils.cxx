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
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
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

}

std::string ROOT::Experimental::Internal::GetCanonicalTypePrefix(const std::string &typeName)
{
   std::string canonicalType{TClassEdit::CleanType(typeName.c_str(), /*mode=*/1)};
   if (canonicalType.substr(0, 7) == "struct ") {
      canonicalType.erase(0, 7);
   } else if (canonicalType.substr(0, 5) == "enum ") {
      canonicalType.erase(0, 5);
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
   std::string normalizedType{GetCanonicalTypePrefix(metaNormalizedName)};
   auto idxOpen = normalizedType.find_first_of("<");
   if (idxOpen == std::string::npos)
      return normalizedType;

   R__ASSERT(normalizedType.back() == '>');
   R__ASSERT((normalizedType.size() - 1) > idxOpen);

   auto templateArgs = TokenizeTypeList(normalizedType.substr(idxOpen + 1, normalizedType.size() - idxOpen - 2));
   R__ASSERT(!templateArgs.empty());

   normalizedType = normalizedType.substr(0, idxOpen + 1); // Everything up to '<'
   for (const auto &a : templateArgs) {
      R__ASSERT(!a.empty());
      if (std::isdigit(a[0]) || a[0] == '-') {
         // Integer template argument
         normalizedType += a + ",";
      } else {
         // Type name template argument; template arguments must keep their CV qualifier
         if (a.substr(0, 6) == "const " || (a.length() > 14 && a.substr(9, 6) == "const "))
            normalizedType += "const ";
         if (a.substr(0, 9) == "volatile " || (a.length() > 14 && a.substr(6, 9) == "volatile "))
            normalizedType += "volatile ";
         normalizedType += GetRenormalizedTypeName(a) + ",";
      }
   }
   normalizedType[normalizedType.size() - 1] = '>';

   return normalizedType;
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

std::tuple<std::string, std::vector<size_t>> ROOT::Experimental::Internal::ParseArrayType(std::string_view typeName)
{
   std::vector<size_t> sizeVec;

   // Only parse outer array definition, i.e. the right `]` should be at the end of the type name
   while (typeName.back() == ']') {
      auto posRBrace = typeName.size() - 1;
      auto posLBrace = typeName.find_last_of('[', posRBrace);
      if (posLBrace == std::string_view::npos)
         return {};

      size_t size;
      if (std::from_chars(typeName.data() + posLBrace + 1, typeName.data() + posRBrace, size).ec != std::errc{})
         return {};
      sizeVec.insert(sizeVec.begin(), size);
      typeName.remove_suffix(typeName.size() - posLBrace);
   }
   return std::make_tuple(std::string{typeName}, sizeVec);
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
