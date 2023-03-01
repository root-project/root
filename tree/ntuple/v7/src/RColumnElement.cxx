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
   case EColumnType::kIndex32: return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kIndex32>>(nullptr);
   case EColumnType::kSwitch: return std::make_unique<RColumnElement<RColumnSwitch, EColumnType::kSwitch>>(nullptr);
   case EColumnType::kByte: return std::make_unique<RColumnElement<std::uint8_t, EColumnType::kByte>>(nullptr);
   case EColumnType::kChar: return std::make_unique<RColumnElement<char, EColumnType::kChar>>(nullptr);
   case EColumnType::kBit: return std::make_unique<RColumnElement<bool, EColumnType::kBit>>(nullptr);
   case EColumnType::kReal64: return std::make_unique<RColumnElement<double, EColumnType::kReal64>>(nullptr);
   case EColumnType::kReal32: return std::make_unique<RColumnElement<float, EColumnType::kReal32>>(nullptr);
   case EColumnType::kInt64: return std::make_unique<RColumnElement<std::int64_t, EColumnType::kInt64>>(nullptr);
   case EColumnType::kInt32: return std::make_unique<RColumnElement<std::int32_t, EColumnType::kInt32>>(nullptr);
   case EColumnType::kInt16: return std::make_unique<RColumnElement<std::int16_t, EColumnType::kInt16>>(nullptr);
   case EColumnType::kInt8: return std::make_unique<RColumnElement<std::int8_t, EColumnType::kInt8>>(nullptr);
   case EColumnType::kSplitIndex32:
      return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kSplitIndex32>>(nullptr);
   case EColumnType::kSplitReal64: return std::make_unique<RColumnElement<double, EColumnType::kSplitReal64>>(nullptr);
   case EColumnType::kSplitReal32: return std::make_unique<RColumnElement<float, EColumnType::kSplitReal32>>(nullptr);
   case EColumnType::kSplitInt64:
      return std::make_unique<RColumnElement<std::int64_t, EColumnType::kSplitInt64>>(nullptr);
   case EColumnType::kSplitInt32:
      return std::make_unique<RColumnElement<std::int32_t, EColumnType::kSplitInt32>>(nullptr);
   case EColumnType::kSplitInt16:
      return std::make_unique<RColumnElement<std::int16_t, EColumnType::kSplitInt16>>(nullptr);
   default: R__ASSERT(false);
   }
   // never here
   return nullptr;
}

std::size_t ROOT::Experimental::Detail::RColumnElementBase::GetBitsOnStorage(EColumnType type) {
   switch (type) {
   case EColumnType::kIndex32: return 32;
   case EColumnType::kSwitch: return 64;
   case EColumnType::kByte: return 8;
   case EColumnType::kChar: return 8;
   case EColumnType::kBit: return 1;
   case EColumnType::kReal64: return 64;
   case EColumnType::kReal32: return 32;
   case EColumnType::kInt64: return 64;
   case EColumnType::kInt32: return 32;
   case EColumnType::kInt16: return 16;
   case EColumnType::kInt8: return 8;
   case EColumnType::kSplitIndex32: return 32;
   case EColumnType::kSplitReal64: return 64;
   case EColumnType::kSplitReal32: return 32;
   case EColumnType::kSplitInt64: return 64;
   case EColumnType::kSplitInt32: return 32;
   case EColumnType::kSplitInt16: return 16;
   default: R__ASSERT(false);
   }
   // never here
   return 0;
}

std::string ROOT::Experimental::Detail::RColumnElementBase::GetTypeName(EColumnType type) {
   switch (type) {
   case EColumnType::kIndex32: return "Index32";
   case EColumnType::kSwitch: return "Switch";
   case EColumnType::kByte: return "Byte";
   case EColumnType::kChar: return "Char";
   case EColumnType::kBit: return "Bit";
   case EColumnType::kReal64: return "Real64";
   case EColumnType::kReal32: return "Real32";
   case EColumnType::kInt64: return "Int64";
   case EColumnType::kInt32: return "Int32";
   case EColumnType::kInt16: return "Int16";
   case EColumnType::kInt8: return "Int8";
   case EColumnType::kSplitIndex32: return "SplitIndex32";
   case EColumnType::kSplitReal64: return "SplitReal64";
   case EColumnType::kSplitReal32: return "SplitReal32";
   case EColumnType::kSplitInt64: return "SplitInt64";
   case EColumnType::kSplitInt32: return "SplitInt32";
   case EColumnType::kSplitInt16: return "SplitInt16";
   default: return "UNKNOWN";
   }
}

void ROOT::Experimental::Detail::RColumnElement<ROOT::Experimental::RColumnSwitch,
                                                ROOT::Experimental::EColumnType::kSwitch>::Pack(void *dst, void *src,
                                                                                                std::size_t count) const
{
   auto srcArray = reinterpret_cast<ROOT::Experimental::RColumnSwitch *>(src);
   auto uint64Array = reinterpret_cast<std::uint64_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      uint64Array[i] =
         (static_cast<std::uint64_t>(srcArray[i].GetTag()) << 44) | (srcArray[i].GetIndex() & 0x0fffffffffff);
#if R__LITTLE_ENDIAN == 0
      uint64Array[i] = RByteSwap<8>::bswap(uint64Array[i]);
#endif
   }
}

void ROOT::Experimental::Detail::RColumnElement<
   ROOT::Experimental::RColumnSwitch, ROOT::Experimental::EColumnType::kSwitch>::Unpack(void *dst, void *src,
                                                                                        std::size_t count) const
{
   auto uint64Array = reinterpret_cast<std::uint64_t *>(src);
   auto dstArray = reinterpret_cast<ROOT::Experimental::RColumnSwitch *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
#if R__LITTLE_ENDIAN == 1
      const auto value = uint64Array[i];
#else
      const auto value = RByteSwap<8>::bswap(uint64Array[i]);
#endif
      dstArray[i] = ROOT::Experimental::RColumnSwitch(
         ClusterSize_t{static_cast<RClusterSize::ValueType>(value & 0x0fffffffffff)}, (value >> 44));
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

void ROOT::Experimental::Detail::RColumnElement<
   ROOT::Experimental::ClusterSize_t, ROOT::Experimental::EColumnType::kIndex32>::Pack(void *dst, void *src,
                                                                                       std::size_t count) const
{
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(src);
   std::int32_t *int32Array = reinterpret_cast<std::int32_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      int32Array[i] = int64Array[i];
#if R__LITTLE_ENDIAN == 0
      int32Array[i] = RByteSwap<4>::bswap(int32Array[i]);
#endif
   }
}

void ROOT::Experimental::Detail::RColumnElement<
   ROOT::Experimental::ClusterSize_t, ROOT::Experimental::EColumnType::kIndex32>::Unpack(void *dst, void *src,
                                                                                         std::size_t count) const
{
   std::int32_t *int32Array = reinterpret_cast<std::int32_t *>(src);
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      int64Array[i] = int32Array[i];
#if R__LITTLE_ENDIAN == 0
      int64Array[i] = RByteSwap<8>::bswap(int64Array[i]);
#endif
   }
}

void ROOT::Experimental::Detail::RColumnElement<
   ROOT::Experimental::ClusterSize_t, ROOT::Experimental::EColumnType::kSplitIndex32>::Pack(void *dst, void *src,
                                                                                            std::size_t count) const
{
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(src);
   char *int32SplitArray = reinterpret_cast<char *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t val = (i == 0) ? int64Array[0] : int64Array[i] - int64Array[i - 1];
#if R__LITTLE_ENDIAN == 0
      val = RByteSwap<4>::bswap(val);
#endif
      for (std::size_t b = 0; b < 4; ++b) {
         int32SplitArray[b * count + i] = reinterpret_cast<char *>(&val)[b];
      }
   }
}

void ROOT::Experimental::Detail::RColumnElement<
   ROOT::Experimental::ClusterSize_t, ROOT::Experimental::EColumnType::kSplitIndex32>::Unpack(void *dst, void *src,
                                                                                              std::size_t count) const
{
   char *int32SplitArray = reinterpret_cast<char *>(src);
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::int32_t v;
      for (std::size_t b = 0; b < 4; ++b) {
         reinterpret_cast<char *>(&v)[b] = int32SplitArray[b * count + i];
      }
#if R__LITTLE_ENDIAN == 0
      v = RByteSwap<4>::bswap(v);
#endif
      int64Array[i] = (i == 0) ? v : int64Array[i - 1] + v;
   }
}

void ROOT::Experimental::Detail::RColumnElement<std::int64_t, ROOT::Experimental::EColumnType::kInt32>::Pack(
  void *dst, void *src, std::size_t count) const
{
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(src);
   std::int32_t *int32Array = reinterpret_cast<std::int32_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      int32Array[i] = int64Array[i];
#if R__LITTLE_ENDIAN == 0
      int32Array[i] = RByteSwap<4>::bswap(int32Array[i]);
#endif
   }
}

void ROOT::Experimental::Detail::RColumnElement<std::int64_t, ROOT::Experimental::EColumnType::kInt32>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   std::int32_t *int32Array = reinterpret_cast<std::int32_t *>(src);
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      int64Array[i] = int32Array[i];
#if R__LITTLE_ENDIAN == 0
      int64Array[i] = RByteSwap<8>::bswap(int64Array[i]);
#endif
   }
}

void ROOT::Experimental::Detail::RColumnElement<std::int64_t, ROOT::Experimental::EColumnType::kSplitInt32>::Pack(
   void *dst, void *src, std::size_t count) const
{
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(src);
   char *int32SplitArray = reinterpret_cast<char *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::int32_t v = int64Array[i];
#if R__LITTLE_ENDIAN == 0
      v = RByteSwap<4>::bswap(v);
#endif
      for (std::size_t b = 0; b < 4; ++b) {
         int32SplitArray[b * count + i] = reinterpret_cast<char *>(&v)[b];
      }
   }
}

void ROOT::Experimental::Detail::RColumnElement<std::int64_t, ROOT::Experimental::EColumnType::kSplitInt32>::Unpack(
   void *dst, void *src, std::size_t count) const
{
   char *int32SplitArray = reinterpret_cast<char *>(src);
   std::int64_t *int64Array = reinterpret_cast<std::int64_t *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::int32_t v = 0;
      for (std::size_t b = 0; b < 4; ++b) {
         reinterpret_cast<char *>(&v)[b] = int32SplitArray[b * count + i];
      }
#if R__LITTLE_ENDIAN == 0
      v = RByteSwap<4>::bswap(v);
#endif
      int64Array[i] = v;
   }
}
