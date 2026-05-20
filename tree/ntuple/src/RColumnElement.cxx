/// \file RColumnElement.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-11

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RColumn.hxx"
#include <ROOT/RColumnElementBase.hxx>

#include "RColumnElement.hxx"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

using ROOT::Internal::RColumnIndex;
using ROOT::Internal::RColumnSwitch;

std::pair<std::uint16_t, std::uint16_t> ROOT::Internal::RColumnElementBase::GetValidBitRange(ENTupleColumnType type)
{
   switch (type) {
   case ENTupleColumnType::kIndex64: return std::make_pair(64, 64);
   case ENTupleColumnType::kIndex32: return std::make_pair(32, 32);
   case ENTupleColumnType::kSwitch: return std::make_pair(96, 96);
   case ENTupleColumnType::kByte: return std::make_pair(8, 8);
   case ENTupleColumnType::kChar: return std::make_pair(8, 8);
   case ENTupleColumnType::kBit: return std::make_pair(1, 1);
   case ENTupleColumnType::kReal64: return std::make_pair(64, 64);
   case ENTupleColumnType::kReal32: return std::make_pair(32, 32);
   case ENTupleColumnType::kReal16: return std::make_pair(16, 16);
   case ENTupleColumnType::kInt64: return std::make_pair(64, 64);
   case ENTupleColumnType::kUInt64: return std::make_pair(64, 64);
   case ENTupleColumnType::kInt32: return std::make_pair(32, 32);
   case ENTupleColumnType::kUInt32: return std::make_pair(32, 32);
   case ENTupleColumnType::kInt16: return std::make_pair(16, 16);
   case ENTupleColumnType::kUInt16: return std::make_pair(16, 16);
   case ENTupleColumnType::kInt8: return std::make_pair(8, 8);
   case ENTupleColumnType::kUInt8: return std::make_pair(8, 8);
   case ENTupleColumnType::kSplitIndex64: return std::make_pair(64, 64);
   case ENTupleColumnType::kSplitIndex32: return std::make_pair(32, 32);
   case ENTupleColumnType::kSplitReal64: return std::make_pair(64, 64);
   case ENTupleColumnType::kSplitReal32: return std::make_pair(32, 32);
   case ENTupleColumnType::kSplitInt64: return std::make_pair(64, 64);
   case ENTupleColumnType::kSplitUInt64: return std::make_pair(64, 64);
   case ENTupleColumnType::kSplitInt32: return std::make_pair(32, 32);
   case ENTupleColumnType::kSplitUInt32: return std::make_pair(32, 32);
   case ENTupleColumnType::kSplitInt16: return std::make_pair(16, 16);
   case ENTupleColumnType::kSplitUInt16: return std::make_pair(16, 16);
   case ENTupleColumnType::kReal32Trunc: return std::make_pair(10, 31);
   case ENTupleColumnType::kReal32Quant: return std::make_pair(1, 32);
   default:
      if (type == kTestFutureColumnType)
         return std::make_pair(32, 32);
      R__ASSERT(false);
   }
   // never here
   return std::make_pair(0, 0);
}

const char *ROOT::Internal::RColumnElementBase::GetColumnTypeName(ENTupleColumnType type)
{
   switch (type) {
   case ENTupleColumnType::kIndex64: return "Index64";
   case ENTupleColumnType::kIndex32: return "Index32";
   case ENTupleColumnType::kSwitch: return "Switch";
   case ENTupleColumnType::kByte: return "Byte";
   case ENTupleColumnType::kChar: return "Char";
   case ENTupleColumnType::kBit: return "Bit";
   case ENTupleColumnType::kReal64: return "Real64";
   case ENTupleColumnType::kReal32: return "Real32";
   case ENTupleColumnType::kReal16: return "Real16";
   case ENTupleColumnType::kInt64: return "Int64";
   case ENTupleColumnType::kUInt64: return "UInt64";
   case ENTupleColumnType::kInt32: return "Int32";
   case ENTupleColumnType::kUInt32: return "UInt32";
   case ENTupleColumnType::kInt16: return "Int16";
   case ENTupleColumnType::kUInt16: return "UInt16";
   case ENTupleColumnType::kInt8: return "Int8";
   case ENTupleColumnType::kUInt8: return "UInt8";
   case ENTupleColumnType::kSplitIndex64: return "SplitIndex64";
   case ENTupleColumnType::kSplitIndex32: return "SplitIndex32";
   case ENTupleColumnType::kSplitReal64: return "SplitReal64";
   case ENTupleColumnType::kSplitReal32: return "SplitReal32";
   case ENTupleColumnType::kSplitInt64: return "SplitInt64";
   case ENTupleColumnType::kSplitUInt64: return "SplitUInt64";
   case ENTupleColumnType::kSplitInt32: return "SplitInt32";
   case ENTupleColumnType::kSplitUInt32: return "SplitUInt32";
   case ENTupleColumnType::kSplitInt16: return "SplitInt16";
   case ENTupleColumnType::kSplitUInt16: return "SplitUInt16";
   case ENTupleColumnType::kReal32Trunc: return "Real32Trunc";
   case ENTupleColumnType::kReal32Quant: return "Real32Quant";
   default:
      if (type == kTestFutureColumnType)
         return "TestFutureType";
      return "UNKNOWN";
   }
}

template <>
std::unique_ptr<ROOT::Internal::RColumnElementBase>
ROOT::Internal::RColumnElementBase::Generate<void>(ENTupleColumnType onDiskType)
{
   //clang-format off
   switch (onDiskType) {
   case ENTupleColumnType::kIndex64: return std::make_unique<RColumnElement<RColumnIndex, ENTupleColumnType::kIndex64>>();
   case ENTupleColumnType::kIndex32: return std::make_unique<RColumnElement<RColumnIndex, ENTupleColumnType::kIndex32>>();
   case ENTupleColumnType::kSwitch: return std::make_unique<RColumnElement<RColumnSwitch, ENTupleColumnType::kSwitch>>();
   case ENTupleColumnType::kByte: return std::make_unique<RColumnElement<std::byte, ENTupleColumnType::kByte>>();
   case ENTupleColumnType::kChar: return std::make_unique<RColumnElement<char, ENTupleColumnType::kChar>>();
   case ENTupleColumnType::kBit: return std::make_unique<RColumnElement<bool, ENTupleColumnType::kBit>>();
   case ENTupleColumnType::kReal64: return std::make_unique<RColumnElement<double, ENTupleColumnType::kReal64>>();
   case ENTupleColumnType::kReal32: return std::make_unique<RColumnElement<float, ENTupleColumnType::kReal32>>();
   // TODO: Change to std::float16_t in-memory type once available (from C++23).
   case ENTupleColumnType::kReal16: return std::make_unique<RColumnElement<float, ENTupleColumnType::kReal16>>();
   case ENTupleColumnType::kInt64: return std::make_unique<RColumnElement<std::int64_t, ENTupleColumnType::kInt64>>();
   case ENTupleColumnType::kUInt64: return std::make_unique<RColumnElement<std::uint64_t, ENTupleColumnType::kUInt64>>();
   case ENTupleColumnType::kInt32: return std::make_unique<RColumnElement<std::int32_t, ENTupleColumnType::kInt32>>();
   case ENTupleColumnType::kUInt32: return std::make_unique<RColumnElement<std::uint32_t, ENTupleColumnType::kUInt32>>();
   case ENTupleColumnType::kInt16: return std::make_unique<RColumnElement<std::int16_t, ENTupleColumnType::kInt16>>();
   case ENTupleColumnType::kUInt16: return std::make_unique<RColumnElement<std::uint16_t, ENTupleColumnType::kUInt16>>();
   case ENTupleColumnType::kInt8: return std::make_unique<RColumnElement<std::int8_t, ENTupleColumnType::kInt8>>();
   case ENTupleColumnType::kUInt8: return std::make_unique<RColumnElement<std::uint8_t, ENTupleColumnType::kUInt8>>();
   case ENTupleColumnType::kSplitIndex64: return std::make_unique<RColumnElement<RColumnIndex, ENTupleColumnType::kSplitIndex64>>();
   case ENTupleColumnType::kSplitIndex32: return std::make_unique<RColumnElement<RColumnIndex, ENTupleColumnType::kSplitIndex32>>();
   case ENTupleColumnType::kSplitReal64: return std::make_unique<RColumnElement<double, ENTupleColumnType::kSplitReal64>>();
   case ENTupleColumnType::kSplitReal32: return std::make_unique<RColumnElement<float, ENTupleColumnType::kSplitReal32>>();
   case ENTupleColumnType::kSplitInt64: return std::make_unique<RColumnElement<std::int64_t, ENTupleColumnType::kSplitInt64>>();
   case ENTupleColumnType::kSplitUInt64: return std::make_unique<RColumnElement<std::uint64_t, ENTupleColumnType::kSplitUInt64>>();
   case ENTupleColumnType::kSplitInt32: return std::make_unique<RColumnElement<std::int32_t, ENTupleColumnType::kSplitInt32>>();
   case ENTupleColumnType::kSplitUInt32: return std::make_unique<RColumnElement<std::uint32_t, ENTupleColumnType::kSplitUInt32>>();
   case ENTupleColumnType::kSplitInt16: return std::make_unique<RColumnElement<std::int16_t, ENTupleColumnType::kSplitInt16>>();
   case ENTupleColumnType::kSplitUInt16: return std::make_unique<RColumnElement<std::uint16_t, ENTupleColumnType::kSplitUInt16>>();
   case ENTupleColumnType::kReal32Trunc: return std::make_unique<RColumnElement<float, ENTupleColumnType::kReal32Trunc>>();
   case ENTupleColumnType::kReal32Quant: return std::make_unique<RColumnElement<float, ENTupleColumnType::kReal32Quant>>();
   default:
      if (onDiskType == kTestFutureColumnType)
         return std::make_unique<RColumnElement<Internal::RTestFutureColumn, kTestFutureColumnType>>();
      R__ASSERT(false);
   }
   //clang-format on
   // never here
   return nullptr;
}

std::unique_ptr<ROOT::Internal::RColumnElementBase>
ROOT::Internal::GenerateColumnElement(std::type_index inMemoryType, ENTupleColumnType onDiskType)
{
   if (inMemoryType == std::type_index(typeid(char))) {
      return GenerateColumnElementInternal<char>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(bool))) {
      return GenerateColumnElementInternal<bool>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::byte))) {
      return GenerateColumnElementInternal<std::byte>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::uint8_t))) {
      return GenerateColumnElementInternal<std::uint8_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::uint16_t))) {
      return GenerateColumnElementInternal<std::uint16_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::uint32_t))) {
      return GenerateColumnElementInternal<std::uint32_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::uint64_t))) {
      return GenerateColumnElementInternal<std::uint64_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::int8_t))) {
      return GenerateColumnElementInternal<std::int8_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::int16_t))) {
      return GenerateColumnElementInternal<std::int16_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::int32_t))) {
      return GenerateColumnElementInternal<std::int32_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(std::int64_t))) {
      return GenerateColumnElementInternal<std::int64_t>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(float))) {
      return GenerateColumnElementInternal<float>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(double))) {
      return GenerateColumnElementInternal<double>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(RColumnIndex))) {
      return GenerateColumnElementInternal<RColumnIndex>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(RColumnSwitch))) {
      return GenerateColumnElementInternal<RColumnSwitch>(onDiskType);
   } else if (inMemoryType == std::type_index(typeid(RTestFutureColumn))) {
      return GenerateColumnElementInternal<RTestFutureColumn>(onDiskType);
   } else {
      R__ASSERT(!"Invalid memory type in GenerateColumnElement");
   }
   // never here
   return nullptr;
}

std::unique_ptr<ROOT::Internal::RColumnElementBase>
ROOT::Internal::GenerateColumnElement(const RColumnElementBase::RIdentifier &elementId)
{
   return GenerateColumnElement(elementId.fInMemoryType, elementId.fOnDiskType);
}

void ROOT::Internal::BitPacking::PackBits(void *dst, const void *src, std::size_t count, std::size_t sizeofSrc,
                                          std::size_t nDstBits)
{
   assert(sizeofSrc <= sizeof(Word_t));
   assert(0 < nDstBits && nDstBits <= sizeofSrc * 8);

   const unsigned char *srcArray = reinterpret_cast<const unsigned char *>(src);
   Word_t *dstArray = reinterpret_cast<Word_t *>(dst);
   Word_t accum = 0;
   std::size_t bitsUsed = 0;
   std::size_t dstIdx = 0;
   for (std::size_t i = 0; i < count; ++i) {
      Word_t packedWord = 0;
      memcpy(&packedWord, srcArray + i * sizeofSrc, sizeofSrc);
      // truncate the LSB of the item
      packedWord >>= sizeofSrc * 8 - nDstBits;

      const std::size_t bitsRem = kBitsPerWord - bitsUsed;
      if (bitsRem >= nDstBits) {
         // append the entire item to the accumulator
         accum |= (packedWord << bitsUsed);
         bitsUsed += nDstBits;
      } else {
         // chop up the item into its `bitsRem` LSB bits + `nDstBits - bitsRem` MSB bits.
         // The LSB bits will be saved in the current word and the MSB will be saved in the next one.
         if (bitsRem > 0) {
            Word_t packedWordLsb = packedWord;
            packedWordLsb <<= (kBitsPerWord - bitsRem);
            packedWordLsb >>= (kBitsPerWord - bitsRem);
            accum |= (packedWordLsb << bitsUsed);
         }

         memcpy(&dstArray[dstIdx++], &accum, sizeof(accum));
         accum = 0;
         bitsUsed = 0;

         if (bitsRem > 0) {
            Word_t packedWordMsb = packedWord;
            packedWordMsb >>= bitsRem;
            accum |= packedWordMsb;
            bitsUsed += nDstBits - bitsRem;
         } else {
            // we realigned to a word boundary: append the entire item
            accum = packedWord;
            bitsUsed += nDstBits;
         }
      }
   }

   if (bitsUsed)
      memcpy(&dstArray[dstIdx++], &accum, (bitsUsed + 7) / 8);

   [[maybe_unused]] auto expDstCount = (count * nDstBits + kBitsPerWord - 1) / kBitsPerWord;
   assert(dstIdx == expDstCount);
}

void ROOT::Internal::BitPacking::UnpackBits(void *dst, const void *src, std::size_t count, std::size_t sizeofDst,
                                            std::size_t nSrcBits)
{
   assert(sizeofDst <= sizeof(Word_t));
   assert(0 < nSrcBits && nSrcBits <= sizeofDst * 8);

   unsigned char *dstArray = reinterpret_cast<unsigned char *>(dst);
   const Word_t *srcArray = reinterpret_cast<const Word_t *>(src);
   const auto nWordsToLoad = (count * nSrcBits + kBitsPerWord - 1) / kBitsPerWord;

   // bit offset of the next packed item inside the currently loaded word
   int offInWord = 0;
   std::size_t dstIdx = 0;
   Word_t prevWordLsb = 0;
   std::size_t remBytesToLoad = (count * nSrcBits + 7) / 8;
   for (std::size_t i = 0; i < nWordsToLoad; ++i) {
      assert(dstIdx < count);

      // load the next word, containing some packed items
      Word_t packedBytes = 0;
      std::size_t bytesLoaded = std::min(remBytesToLoad, sizeof(Word_t));
      memcpy(&packedBytes, &srcArray[i], bytesLoaded);

      assert(remBytesToLoad >= bytesLoaded);
      remBytesToLoad -= bytesLoaded;

      // If `offInWord` is negative, it means that the last item was split
      // across 2 words and we need to recombine it.
      if (offInWord < 0) {
         std::size_t nMsb = nSrcBits + offInWord;
         std::uint32_t msb = packedBytes << (8 * sizeofDst - nMsb);
         Word_t packedWord = msb | prevWordLsb;
         prevWordLsb = 0;
         memcpy(dstArray + dstIdx * sizeofDst, &packedWord, sizeofDst);
         ++dstIdx;
         offInWord = nMsb;
      }

      // isolate each item in the loaded word
      while (dstIdx < count) {
         // Check if we need to load a split item or a full one
         if (offInWord > static_cast<int>(kBitsPerWord - nSrcBits)) {
            // save the LSB of the next item, next `for` loop will merge them with the MSB in the next word.
            assert(offInWord <= static_cast<int>(kBitsPerWord));
            std::size_t nLsbNext = kBitsPerWord - offInWord;
            if (nLsbNext)
               prevWordLsb = (packedBytes >> offInWord) << (8 * sizeofDst - nSrcBits);
            offInWord -= kBitsPerWord;
            break;
         }

         Word_t packedWord = packedBytes;
         assert(nSrcBits + offInWord <= kBitsPerWord);
         packedWord >>= offInWord;
         packedWord <<= 8 * sizeofDst - nSrcBits;
         memcpy(dstArray + dstIdx * sizeofDst, &packedWord, sizeofDst);
         ++dstIdx;
         offInWord += nSrcBits;
      }
   }

   assert(prevWordLsb == 0);
   assert(dstIdx == count);
}
