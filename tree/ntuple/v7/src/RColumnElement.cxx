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

#include "ROOT/RColumn.hxx"
#include <ROOT/RColumnElementBase.hxx>

#include "RColumnElement.hxx"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

std::pair<std::uint16_t, std::uint16_t>
ROOT::Experimental::Internal::RColumnElementBase::GetValidBitRange(EColumnType type)
{
   switch (type) {
   case EColumnType::kIndex64: return std::make_pair(64, 64);
   case EColumnType::kIndex32: return std::make_pair(32, 32);
   case EColumnType::kSwitch: return std::make_pair(96, 96);
   case EColumnType::kByte: return std::make_pair(8, 8);
   case EColumnType::kChar: return std::make_pair(8, 8);
   case EColumnType::kBit: return std::make_pair(1, 1);
   case EColumnType::kReal64: return std::make_pair(64, 64);
   case EColumnType::kReal32: return std::make_pair(32, 32);
   case EColumnType::kReal16: return std::make_pair(16, 16);
   case EColumnType::kInt64: return std::make_pair(64, 64);
   case EColumnType::kUInt64: return std::make_pair(64, 64);
   case EColumnType::kInt32: return std::make_pair(32, 32);
   case EColumnType::kUInt32: return std::make_pair(32, 32);
   case EColumnType::kInt16: return std::make_pair(16, 16);
   case EColumnType::kUInt16: return std::make_pair(16, 16);
   case EColumnType::kInt8: return std::make_pair(8, 8);
   case EColumnType::kUInt8: return std::make_pair(8, 8);
   case EColumnType::kSplitIndex64: return std::make_pair(64, 64);
   case EColumnType::kSplitIndex32: return std::make_pair(32, 32);
   case EColumnType::kSplitReal64: return std::make_pair(64, 64);
   case EColumnType::kSplitReal32: return std::make_pair(32, 32);
   case EColumnType::kSplitInt64: return std::make_pair(64, 64);
   case EColumnType::kSplitUInt64: return std::make_pair(64, 64);
   case EColumnType::kSplitInt32: return std::make_pair(32, 32);
   case EColumnType::kSplitUInt32: return std::make_pair(32, 32);
   case EColumnType::kSplitInt16: return std::make_pair(16, 16);
   case EColumnType::kSplitUInt16: return std::make_pair(16, 16);
   case EColumnType::kReal32Trunc: return std::make_pair(kReal32TruncBitsMin, kReal32TruncBitsMax);
   default: assert(false);
   }
   // never here
   return std::make_pair(0, 0);
}

std::string ROOT::Experimental::Internal::RColumnElementBase::GetTypeName(EColumnType type)
{
   switch (type) {
   case EColumnType::kIndex64: return "Index64";
   case EColumnType::kIndex32: return "Index32";
   case EColumnType::kSwitch: return "Switch";
   case EColumnType::kByte: return "Byte";
   case EColumnType::kChar: return "Char";
   case EColumnType::kBit: return "Bit";
   case EColumnType::kReal64: return "Real64";
   case EColumnType::kReal32: return "Real32";
   case EColumnType::kReal16: return "Real16";
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
   case EColumnType::kReal32Trunc: return "Real32Trunc";
   default: return "UNKNOWN";
   }
}

template <>
std::unique_ptr<ROOT::Experimental::Internal::RColumnElementBase>
ROOT::Experimental::Internal::RColumnElementBase::Generate<void>(EColumnType type)
{
   switch (type) {
   case EColumnType::kIndex64: return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kIndex64>>();
   case EColumnType::kIndex32: return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kIndex32>>();
   case EColumnType::kSwitch: return std::make_unique<RColumnElement<RColumnSwitch, EColumnType::kSwitch>>();
   case EColumnType::kByte: return std::make_unique<RColumnElement<std::byte, EColumnType::kByte>>();
   case EColumnType::kChar: return std::make_unique<RColumnElement<char, EColumnType::kChar>>();
   case EColumnType::kBit: return std::make_unique<RColumnElement<bool, EColumnType::kBit>>();
   case EColumnType::kReal64: return std::make_unique<RColumnElement<double, EColumnType::kReal64>>();
   case EColumnType::kReal32: return std::make_unique<RColumnElement<float, EColumnType::kReal32>>();
   // TODO: Change to std::float16_t in-memory type once available (from C++23).
   case EColumnType::kReal16: return std::make_unique<RColumnElement<float, EColumnType::kReal16>>();
   case EColumnType::kInt64: return std::make_unique<RColumnElement<std::int64_t, EColumnType::kInt64>>();
   case EColumnType::kUInt64: return std::make_unique<RColumnElement<std::uint64_t, EColumnType::kUInt64>>();
   case EColumnType::kInt32: return std::make_unique<RColumnElement<std::int32_t, EColumnType::kInt32>>();
   case EColumnType::kUInt32: return std::make_unique<RColumnElement<std::uint32_t, EColumnType::kUInt32>>();
   case EColumnType::kInt16: return std::make_unique<RColumnElement<std::int16_t, EColumnType::kInt16>>();
   case EColumnType::kUInt16: return std::make_unique<RColumnElement<std::uint16_t, EColumnType::kUInt16>>();
   case EColumnType::kInt8: return std::make_unique<RColumnElement<std::int8_t, EColumnType::kInt8>>();
   case EColumnType::kUInt8: return std::make_unique<RColumnElement<std::uint8_t, EColumnType::kUInt8>>();
   case EColumnType::kSplitIndex64:
      return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kSplitIndex64>>();
   case EColumnType::kSplitIndex32:
      return std::make_unique<RColumnElement<ClusterSize_t, EColumnType::kSplitIndex32>>();
   case EColumnType::kSplitReal64: return std::make_unique<RColumnElement<double, EColumnType::kSplitReal64>>();
   case EColumnType::kSplitReal32: return std::make_unique<RColumnElement<float, EColumnType::kSplitReal32>>();
   case EColumnType::kSplitInt64: return std::make_unique<RColumnElement<std::int64_t, EColumnType::kSplitInt64>>();
   case EColumnType::kSplitUInt64: return std::make_unique<RColumnElement<std::uint64_t, EColumnType::kSplitUInt64>>();
   case EColumnType::kSplitInt32: return std::make_unique<RColumnElement<std::int32_t, EColumnType::kSplitInt32>>();
   case EColumnType::kSplitUInt32: return std::make_unique<RColumnElement<std::uint32_t, EColumnType::kSplitUInt32>>();
   case EColumnType::kSplitInt16: return std::make_unique<RColumnElement<std::int16_t, EColumnType::kSplitInt16>>();
   case EColumnType::kSplitUInt16: return std::make_unique<RColumnElement<std::uint16_t, EColumnType::kSplitUInt16>>();
   case EColumnType::kReal32Trunc: return std::make_unique<RColumnElement<std::uint16_t, EColumnType::kReal32Trunc>>();
   default: assert(false);
   }
   // never here
   return nullptr;
}

std::unique_ptr<ROOT::Experimental::Internal::RColumnElementBase>
ROOT::Experimental::Internal::GenerateColumnElement(EColumnCppType cppType, EColumnType type)
{
   switch (cppType) {
   case EColumnCppType::kChar: return GenerateColumnElementInternal<char>(type);
   case EColumnCppType::kBool: return GenerateColumnElementInternal<bool>(type);
   case EColumnCppType::kByte: return GenerateColumnElementInternal<std::byte>(type);
   case EColumnCppType::kUint8: return GenerateColumnElementInternal<std::uint8_t>(type);
   case EColumnCppType::kUint16: return GenerateColumnElementInternal<std::uint16_t>(type);
   case EColumnCppType::kUint32: return GenerateColumnElementInternal<std::uint32_t>(type);
   case EColumnCppType::kUint64: return GenerateColumnElementInternal<std::uint64_t>(type);
   case EColumnCppType::kInt8: return GenerateColumnElementInternal<std::int8_t>(type);
   case EColumnCppType::kInt16: return GenerateColumnElementInternal<std::int16_t>(type);
   case EColumnCppType::kInt32: return GenerateColumnElementInternal<std::int32_t>(type);
   case EColumnCppType::kInt64: return GenerateColumnElementInternal<std::int64_t>(type);
   case EColumnCppType::kFloat: return GenerateColumnElementInternal<float>(type);
   case EColumnCppType::kDouble: return GenerateColumnElementInternal<double>(type);
   case EColumnCppType::kClusterSize: return GenerateColumnElementInternal<ClusterSize_t>(type);
   case EColumnCppType::kColumnSwitch: return GenerateColumnElementInternal<RColumnSwitch>(type);
   default: R__ASSERT(!"Invalid column cpp type");
   }
   // never here
   return nullptr;
}

void ROOT::Experimental::Internal::BitPacking::PackBits(void *dst, const void *src, std::size_t count,
                                                        std::size_t sizeofSrc, std::size_t nDstBits)
{
   R__ASSERT(sizeofSrc <= sizeof(Word_t));
   R__ASSERT(0 < nDstBits && nDstBits <= sizeofSrc * 8);
   
   const unsigned char *srcArray = reinterpret_cast<const unsigned char *>(src);
   Word_t *dstArray = reinterpret_cast<Word_t *>(dst);
   Word_t accum = 0;
   std::size_t bitsUsed = 0;
   std::size_t dstIdx = 0;
   for (std::size_t i = 0; i < count; ++i) {
      Word_t packedWord = 0;
      memcpy(&packedWord, srcArray + i * sizeofSrc, sizeofSrc);
      // make sure we represent the packed word as LE on disk
      ByteSwapIfNecessary(packedWord);
      // truncate the LSB of the mantissa
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
      memcpy(&dstArray[dstIdx++], &accum, sizeof(accum));

   [[maybe_unused]] auto expDstCount = (count * nDstBits + kBitsPerWord - 1) / kBitsPerWord;
   assert(dstIdx == expDstCount);
}

void ROOT::Experimental::Internal::BitPacking::UnpackBits(void *dst, const void *src, std::size_t count,
                                                          std::size_t sizeofDst, std::size_t nSrcBits)
{
   R__ASSERT(sizeofDst <= sizeof(Word_t));
   R__ASSERT(0 < nSrcBits && nSrcBits <= sizeofDst * 8);

   unsigned char *dstArray = reinterpret_cast<unsigned char *>(dst);
   const Word_t *srcArray = reinterpret_cast<const Word_t *>(src);
   const auto nWordsToLoad = (count * nSrcBits + kBitsPerWord - 1) / kBitsPerWord;

   // bit offset of the next packed item inside the currently loaded word
   int offInWord = 0;
   std::size_t dstIdx = 0;
   std::uint32_t prevWordAccum = 0;
   for (std::size_t i = 0; i < nWordsToLoad; ++i) {
      assert(dstIdx < count);

      // load the next word, containing some packed items
      Word_t packedBytes;
      memcpy(&packedBytes, &srcArray[i], sizeof(packedBytes));

      // If `offInWord` is negative, it means that the last item was split
      // across 2 words and we need to recombine it.
      if (offInWord < 0) {
         std::size_t nMsb = nSrcBits + offInWord;
         std::uint32_t msb = packedBytes << (8 * sizeofDst - nMsb);
         std::uint32_t packedWord = msb | prevWordAccum;
         prevWordAccum = 0;
         memcpy(dstArray + dstIdx * sizeofDst, &packedWord, sizeofDst);
         ByteSwapIfNecessary(dst[dstIdx]);
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
               prevWordAccum = (packedBytes >> offInWord) << (8 * sizeofDst - nSrcBits);
            offInWord -= kBitsPerWord;
            break;
         }

         Word_t packedWord = packedBytes;
         assert(nSrcBits + offInWord <= kBitsPerWord);
         packedWord >>= offInWord;
         packedWord <<= 8 * sizeofDst - nSrcBits;
         memcpy(dstArray + dstIdx * sizeofDst, &packedWord, sizeofDst);
         ByteSwapIfNecessary(dst[dstIdx]);
         ++dstIdx;
         offInWord += nSrcBits;
      }
   }

   assert(prevWordAccum == 0);
   assert(dstIdx == count);
}
