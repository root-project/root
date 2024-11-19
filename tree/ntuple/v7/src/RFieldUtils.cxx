/// \file RFieldUtils.cxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include "RFieldUtils.hxx"

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

std::string ROOT::Experimental::Internal::GetNormalizedTypeName(const std::string &typeName)
{
   std::string normalizedType{TClassEdit::CleanType(typeName.c_str(), /*mode=*/2)};

   if (auto it = typeTranslationMap.find(normalizedType); it != typeTranslationMap.end())
      normalizedType = it->second;

   if (normalizedType.substr(0, 7) == "vector<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 6) == "array<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 8) == "variant<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 5) == "pair<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 6) == "tuple<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 7) == "bitset<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 11) == "unique_ptr<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 4) == "set<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 14) == "unordered_set<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 9) == "multiset<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 19) == "unordered_multiset<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 4) == "map<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 14) == "unordered_map<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 9) == "multimap<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 19) == "unordered_multimap<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 7) == "atomic<")
      normalizedType = "std::" + normalizedType;

   if (normalizedType.substr(0, 11) == "ROOT::RVec<")
      normalizedType = "ROOT::VecOps::RVec<" + normalizedType.substr(11);

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
      R__LOG_WARNING(ROOT::Experimental::NTupleLog()) << "invalid setting for 'rntuple.streamerMode' class attribute: "
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
