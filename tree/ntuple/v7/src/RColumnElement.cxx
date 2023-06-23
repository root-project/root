/// \file RColumnElement.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RColumnElement.hxx>

#include <TError.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <memory>
#include <utility>

template <>
std::unique_ptr<ROOT::Experimental::Detail::RColumnElementBase>
ROOT::Experimental::Detail::RColumnElementBase::Generate<void>(EColumnType type)
{
   switch (type) {
   case EColumnType::kIndex64: return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kIndex64>>(nullptr);
   case EColumnType::kIndex32: return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kIndex32>>(nullptr);
   case EColumnType::kSwitch: return std::make_unique<RColumnElement<RColumnSwitch, EColumnType::kSwitch>>(nullptr);
   case EColumnType::kByte: return std::make_unique<RColumnElement<std::uint8_t, EColumnType::kByte>>(nullptr);
   case EColumnType::kChar: return std::make_unique<RColumnElement<char, EColumnType::kChar>>(nullptr);
   case EColumnType::kBit: return std::make_unique<RColumnElement<bool, EColumnType::kBit>>(nullptr);
   case EColumnType::kReal64: return std::make_unique<RColumnElement<double, EColumnType::kReal64>>(nullptr);
   case EColumnType::kReal32: return std::make_unique<RColumnElement<float, EColumnType::kReal32>>(nullptr);
   case EColumnType::kInt64: return std::make_unique<RColumnElement<std::int64_t, EColumnType::kInt64>>(nullptr);
   case EColumnType::kUInt64: return std::make_unique<RColumnElement<std::uint64_t, EColumnType::kUInt64>>(nullptr);
   case EColumnType::kInt32: return std::make_unique<RColumnElement<std::int32_t, EColumnType::kInt32>>(nullptr);
   case EColumnType::kUInt32: return std::make_unique<RColumnElement<std::uint32_t, EColumnType::kUInt32>>(nullptr);
   case EColumnType::kInt16: return std::make_unique<RColumnElement<std::int16_t, EColumnType::kInt16>>(nullptr);
   case EColumnType::kUInt16: return std::make_unique<RColumnElement<std::uint16_t, EColumnType::kUInt16>>(nullptr);
   case EColumnType::kInt8: return std::make_unique<RColumnElement<std::int8_t, EColumnType::kInt8>>(nullptr);
   case EColumnType::kUInt8: return std::make_unique<RColumnElement<std::uint8_t, EColumnType::kUInt8>>(nullptr);
   case EColumnType::kSplitIndex64:
      return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kSplitIndex64>>(nullptr);
   case EColumnType::kSplitIndex32:
      return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kSplitIndex32>>(nullptr);
   case EColumnType::kSplitReal64: return std::make_unique<RColumnElement<double, EColumnType::kSplitReal64>>(nullptr);
   case EColumnType::kSplitReal32: return std::make_unique<RColumnElement<float, EColumnType::kSplitReal32>>(nullptr);
   case EColumnType::kSplitInt64:
      return std::make_unique<RColumnElement<std::int64_t, EColumnType::kSplitInt64>>(nullptr);
   case EColumnType::kSplitUInt64:
      return std::make_unique<RColumnElement<std::uint64_t, EColumnType::kSplitUInt64>>(nullptr);
   case EColumnType::kSplitInt32:
      return std::make_unique<RColumnElement<std::int32_t, EColumnType::kSplitInt32>>(nullptr);
   case EColumnType::kSplitUInt32:
      return std::make_unique<RColumnElement<std::uint32_t, EColumnType::kSplitUInt32>>(nullptr);
   case EColumnType::kSplitInt16:
      return std::make_unique<RColumnElement<std::int16_t, EColumnType::kSplitInt16>>(nullptr);
   case EColumnType::kSplitUInt16:
      return std::make_unique<RColumnElement<std::uint16_t, EColumnType::kSplitUInt16>>(nullptr);
   default: R__ASSERT(false);
   }
   // never here
   return nullptr;
}

std::size_t ROOT::Experimental::Detail::RColumnElementBase::GetBitsOnStorage(EColumnType type) {
   switch (type) {
   case EColumnType::kIndex64: return 64;
   case EColumnType::kIndex32: return 32;
   case EColumnType::kSwitch: return 64;
   case EColumnType::kByte: return 8;
   case EColumnType::kChar: return 8;
   case EColumnType::kBit: return 1;
   case EColumnType::kReal64: return 64;
   case EColumnType::kReal32: return 32;
   case EColumnType::kInt64: return 64;
   case EColumnType::kUInt64: return 64;
   case EColumnType::kInt32: return 32;
   case EColumnType::kUInt32: return 32;
   case EColumnType::kInt16: return 16;
   case EColumnType::kUInt16: return 16;
   case EColumnType::kInt8: return 8;
   case EColumnType::kUInt8: return 8;
   case EColumnType::kSplitIndex64: return 64;
   case EColumnType::kSplitIndex32: return 32;
   case EColumnType::kSplitReal64: return 64;
   case EColumnType::kSplitReal32: return 32;
   case EColumnType::kSplitInt64: return 64;
   case EColumnType::kSplitUInt64: return 64;
   case EColumnType::kSplitInt32: return 32;
   case EColumnType::kSplitUInt32: return 32;
   case EColumnType::kSplitInt16: return 16;
   case EColumnType::kSplitUInt16: return 16;
   default: R__ASSERT(false);
   }
   // never here
   return 0;
}

std::string ROOT::Experimental::Detail::RColumnElementBase::GetTypeName(EColumnType type) {
   switch (type) {
   case EColumnType::kIndex64: return "Index64";
   case EColumnType::kIndex32: return "Index32";
   case EColumnType::kSwitch: return "Switch";
   case EColumnType::kByte: return "Byte";
   case EColumnType::kChar: return "Char";
   case EColumnType::kBit: return "Bit";
   case EColumnType::kReal64: return "Real64";
   case EColumnType::kReal32: return "Real32";
   case EColumnType::kInt64: return "Int64";
   case EColumnType::kUInt64: return "UInt64";
   case EColumnType::kInt32: return "Int32";
   case EColumnType::kUInt32: return "UInt32";
   case EColumnType::kInt16: return "Int16";
   case EColumnType::kUInt16: return "UInt16";
   case EColumnType::kInt8: return "Int8";
   case EColumnType::kUInt8: return "UInt8";
   case EColumnType::kSplitIndex64: return "SplitIndex64";
   case EColumnType::kSplitIndex32: return "SplitIndex32";
   case EColumnType::kSplitReal64: return "SplitReal64";
   case EColumnType::kSplitReal32: return "SplitReal32";
   case EColumnType::kSplitInt64: return "SplitInt64";
   case EColumnType::kSplitUInt64: return "SplitUInt64";
   case EColumnType::kSplitInt32: return "SplitInt32";
   case EColumnType::kSplitUInt32: return "SplitUInt32";
   case EColumnType::kSplitInt16: return "SplitInt16";
   case EColumnType::kSplitUInt16: return "SplitUInt16";
   default: return "UNKNOWN";
   }
}

void ROOT::Experimental::Detail::RColumnElement<bool, ROOT::Experimental::EColumnType::kBit>::Pack(
  void *dst, void *src, std::size_t count) const
{
   bool *boolArray = reinterpret_cast<bool *>(src);
   char *charArray = reinterpret_cast<char *>(dst);
   std::bitset<8> bitSet;
   std::size_t i = 0;
   for (; i < count; ++i) {
      bitSet.set(i % 8, boolArray[i]);
      if (i % 8 == 7) {
         char packed = bitSet.to_ulong();
         charArray[i / 8] = packed;
      }
   }
   if (i % 8 != 0) {
      char packed = bitSet.to_ulong();
      charArray[i / 8] = packed;
   }
}

void ROOT::Experimental::Detail::RColumnElement<bool, ROOT::Experimental::EColumnType::kBit>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   bool *boolArray = reinterpret_cast<bool *>(dst);
   char *charArray = reinterpret_cast<char *>(src);
   std::bitset<8> bitSet;
   for (std::size_t i = 0; i < count; i += 8) {
      bitSet = charArray[i / 8];
      for (std::size_t j = i; j < std::min(count, i + 8); ++j) {
         boolArray[j] = bitSet[j % 8];
      }
   }
}
