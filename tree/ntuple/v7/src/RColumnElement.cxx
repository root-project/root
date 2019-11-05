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
#include <limits>

ROOT::Experimental::Detail::RColumnElementBase
ROOT::Experimental::Detail::RColumnElementBase::Generate(EColumnType type) {
   switch (type) {
   case EColumnType::kReal32:
      return RColumnElement<float, EColumnType::kReal32>(nullptr);
   case EColumnType::kReal64:
      return RColumnElement<double, EColumnType::kReal64>(nullptr);
   case EColumnType::kByte:
      return RColumnElement<std::uint8_t, EColumnType::kByte>(nullptr);
   case EColumnType::kInt32:
      return RColumnElement<std::int32_t, EColumnType::kInt32>(nullptr);
   case EColumnType::kInt64:
      return RColumnElement<std::int64_t, EColumnType::kInt64>(nullptr);
   case EColumnType::kBit:
      return RColumnElement<bool, EColumnType::kBit>(nullptr);
   case EColumnType::kIndex:
      return RColumnElement<ClusterSize_t, EColumnType::kIndex>(nullptr);
   case EColumnType::kSwitch:
      return RColumnElement<RColumnSwitch, EColumnType::kSwitch>(nullptr);
   case EColumnType::kReal24:
      return RColumnElement<float, EColumnType::kReal24>(nullptr);
   case EColumnType::kReal16:
      return RColumnElement<float, EColumnType::kReal16>(nullptr);
   case EColumnType::kReal8:
      return RColumnElement<float, EColumnType::kReal8>(nullptr);
   case EColumnType::kCustomDouble:
      return RColumnElement<double, EColumnType::kCustomDouble>(nullptr);
   case EColumnType::kCustomFloat:
      return RColumnElement<float, EColumnType::kCustomFloat>(nullptr);
   default:
      R__ASSERT(false);
   }
   // never here
   return RColumnElementBase();
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

// Way to pack: 1 sign bit, 7 exponent bits, 16 mantissa bits. (same as in Radeon R300 and R420)
void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal24>::Pack(
  void *dst, void *src, std::size_t count) const
{
   R__ASSERT(sizeof(float) == 4);
   std::uint32_t *floatArray = reinterpret_cast<std::uint32_t *>(src);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::bitset<32> floatFlags{ floatArray[i] };
      std::bitset<8> flags{ 0 };
      if (floatFlags.test(31)) flags.set(7);
      if (floatFlags.test(30)) flags.set(6);
      if (floatFlags.test(28)) flags.set(5);
      if (floatFlags.test(27)) flags.set(4);
      if (floatFlags.test(26)) flags.set(3);
      if (floatFlags.test(25)) flags.set(2);
      if (floatFlags.test(24)) flags.set(1);
      if (floatFlags.test(23)) flags.set(0);
      charArray[3*i] = flags.to_ulong();
      charArray[3*i+1] = floatArray[i] >> 15;
      charArray[3*i+2] = floatArray[i] >> 7;
   }
}

void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal24>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   R__ASSERT(sizeof(float) == 4);
   std::uint32_t *floatArray = reinterpret_cast<std::uint32_t *>(dst);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(src);
   for (std::size_t i = 0; i < count; ++i) {
      std::bitset<32> floatFlags{ 0 };
      std::bitset<8> flags{ static_cast<unsigned long long>(charArray[3*i]) };
      std::bitset<7> exponent{ static_cast<unsigned long long>(charArray[3*i]) };
      if (flags.test(0)) floatFlags.set(23);
      if (flags.test(1)) floatFlags.set(24);
      if (flags.test(2)) floatFlags.set(25);
      if (flags.test(3)) floatFlags.set(26);
      if (flags.test(4)) floatFlags.set(27);
      if (flags.test(5)) floatFlags.set(28);
      if (flags.test(6)) floatFlags.set(30);
         else floatFlags.set(29);
      if (flags.test(7)) floatFlags.set(31);
      if (exponent.all()) floatFlags.set(29); // infinity and NaN case
         else if (exponent.none()) floatFlags.reset(29); // zero and denormalized number case
      floatArray[i] = floatFlags.to_ulong() + charArray[3*i+1]*256*128 + charArray[3*i+2]*128;
   }
}

// Way to pack: 1 sign bit, 5 exponent bits, 10 mantissa bits (as in IEE 754-2008)
void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal16>::Pack(
  void *dst, void *src, std::size_t count) const
{
   R__ASSERT(sizeof(float) == 4);
   std::uint32_t *floatArray = reinterpret_cast<std::uint32_t *>(src);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::bitset<32> floatFlags{ floatArray[i] };
      std::bitset<8> flags{ 0 };
      if (floatFlags.test(31)) flags.set(7);
      if (floatFlags.test(30)) flags.set(6);
      if (floatFlags.test(26)) flags.set(5);
      if (floatFlags.test(25)) flags.set(4);
      if (floatFlags.test(24)) flags.set(3);
      if (floatFlags.test(23)) flags.set(2);
      if (floatFlags.test(22)) flags.set(1); // first mantissa bit
      if (floatFlags.test(21)) flags.set(0); // second mantissa bit
      charArray[2*i] = flags.to_ulong();
      charArray[2*i+1] = floatArray[i] >> 13;
   }
}

void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal16>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   R__ASSERT(sizeof(float) == 4);
   std::uint32_t *floatArray = reinterpret_cast<std::uint32_t *>(dst);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(src);
   for (std::size_t i = 0; i < count; ++i) {
      std::bitset<32> floatFlags{ 0 };
      std::bitset<8> flags{ static_cast<unsigned long long>(charArray[2*i]) };
      std::bitset<5> exponent{ static_cast<unsigned long long>(charArray[2*i] >> 2) };
      if (flags.test(0)) floatFlags.set(21);
      if (flags.test(1)) floatFlags.set(22);
      if (flags.test(2)) floatFlags.set(23);
      if (flags.test(3)) floatFlags.set(24);
      if (flags.test(4)) floatFlags.set(25);
      if (flags.test(5)) floatFlags.set(26);
      if (flags.test(6)) floatFlags.set(30);
         else { floatFlags.set(29); floatFlags.set(28); floatFlags.set(27); }
      if (flags.test(7)) floatFlags.set(31);
      if (exponent.all()) { floatFlags.set(29); floatFlags.set(28); floatFlags.set(27); } // infinity and NaN case
         else if (exponent.none()) { floatFlags.reset(29); floatFlags.reset(28); floatFlags.reset(27); } // zero and denormalized number case
      floatArray[i] = floatFlags.to_ulong() + charArray[2*i+1]*256*32;
   }
}

// Way to pack: 1 sign bit, 3 exponent bits, 4 mantissa bits (same as in G.711)
void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal8>::Pack(
  void *dst, void *src, std::size_t count) const
{
   std::uint32_t *floatArray = reinterpret_cast<std::uint32_t *>(src);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(dst);
   for (std::size_t i = 0; i < count; ++i) {
      std::bitset<32> floatFlags{ floatArray[i] };
      std::bitset<8> flags{ 0 };
      if (floatFlags.test(31)) flags.set(7);
      if (floatFlags.test(30)) flags.set(6);
      if (floatFlags.test(24)) flags.set(5);
      if (floatFlags.test(23)) flags.set(4);
      if (floatFlags.test(22)) flags.set(3); // first mantissa bit
      if (floatFlags.test(21)) flags.set(2); // second mantissa bit
      if (floatFlags.test(20)) flags.set(1); // third mantissa bit
      if (floatFlags.test(19)) flags.set(0); // fourth mantissa bit
      charArray[i] = flags.to_ulong();
   }
}

void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kReal8>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   R__ASSERT(sizeof(float) == 4);
   std::uint32_t *floatArray = reinterpret_cast<std::uint32_t *>(dst);
   char *charArray = reinterpret_cast<char *>(src);
   for (std::size_t i = 0; i < count; ++i) {
      std::bitset<32> floatFlags{ 0 };
      std::bitset<8> flags{ static_cast<unsigned long long>(charArray[i]) };
      std::bitset<3> exponent{ static_cast<unsigned long long>(charArray[i] >> 4) };
      if (flags.test(0)) floatFlags.set(19);
      if (flags.test(1)) floatFlags.set(20);
      if (flags.test(2)) floatFlags.set(21);
      if (flags.test(3)) floatFlags.set(22);
      if (flags.test(4)) floatFlags.set(23);
      if (flags.test(5)) floatFlags.set(24);
      if (flags.test(6)) floatFlags.set(30);
      else { floatFlags.set(29); floatFlags.set(28); floatFlags.set(27); floatFlags.set(26); floatFlags.set(25); }
      if (flags.test(7)) floatFlags.set(31);
      if (exponent.all()) { floatFlags.set(29); floatFlags.set(28); floatFlags.set(27); floatFlags.set(26); floatFlags.set(25); } // infinity and NaN case
      else if (exponent.none()) { floatFlags.reset(29); floatFlags.reset(28); floatFlags.reset(27); floatFlags.reset(26); floatFlags.reset(25); } // zero and denormalized number case
      floatArray[i] = floatFlags.to_ulong();
   }
}

void ROOT::Experimental::Detail::RColumnElement<double, ROOT::Experimental::EColumnType::kCustomDouble>::Pack(
  void *dst, void *src, std::size_t count) const
{
   double *doubleArray = reinterpret_cast<double *>(src);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(dst);
   if (kBitsOnStorage % 8 == 0) { // use a faster procedure when %8 == 0
      // check here for inf, nan, overflow
      int numCharUsedToStoreOneValue = kBitsOnStorage / 8;
      for (std::size_t i = 0; i < count; ++i) {
         std::size_t totalNumSteps;
         if (std::isnan(doubleArray[i])) totalNumSteps = 2; // NaN
         else if (doubleArray[i] > fMax) totalNumSteps = 0; // overflow/positive infinity
         else if (doubleArray[i] < fMin) totalNumSteps = 1; // underflow/negative infinity
         else totalNumSteps = std::llround( (doubleArray[i]-fMin) / fStep) + 3;
         for (int j = 0; j < numCharUsedToStoreOneValue; ++j) {
            charArray[numCharUsedToStoreOneValue*i+j] = (totalNumSteps & 0xFF);
            totalNumSteps >>= 8;
         }
      }
   } else {
      int index = 0; // of charArray, count is index of doubleArray.
      short bitsFilledInCurrentCharArrayUpToNow = 0; // takes values from 0 to 7
      for (std::size_t i = 0; i < count; ++i) {
         short unfilledNumBitsOfCurrentValue = kBitsOnStorage;
         std::size_t totalNumSteps;
         if (std::isnan(doubleArray[i])) totalNumSteps = 2; // NaN
         else if (doubleArray[i] > fMax) totalNumSteps = 0; // overflow/positive infinity
         else if (doubleArray[i] < fMin) totalNumSteps = 1; // underflow/negative infinity
         else totalNumSteps = std::llround( (doubleArray[i]-fMin) / fStep) + 3;
         if (bitsFilledInCurrentCharArrayUpToNow != 0) {
            std::size_t nBitsToFill = std::min((int)kBitsOnStorage, (8 - bitsFilledInCurrentCharArrayUpToNow));
            short mask = (1 << nBitsToFill)-1;
            unsigned char ToAdd {static_cast<unsigned char>(totalNumSteps & mask)};
            ToAdd <<= (8 - bitsFilledInCurrentCharArrayUpToNow - nBitsToFill);
            charArray[index] += ToAdd; //>>= nBitsToFill);
            totalNumSteps >>= nBitsToFill;
            bitsFilledInCurrentCharArrayUpToNow += nBitsToFill;
            R__ASSERT(bitsFilledInCurrentCharArrayUpToNow <= 8);
            unfilledNumBitsOfCurrentValue -= nBitsToFill;
            if (bitsFilledInCurrentCharArrayUpToNow == 8) {
               ++index;
               bitsFilledInCurrentCharArrayUpToNow = 0;
            } else {
               continue;
            }
         }
         // Fill where entire 8 bits can be filled at once, with while loop instead of a for loop.
         while(unfilledNumBitsOfCurrentValue >= 8) {
            charArray[index] = (totalNumSteps & 0xFF);
            totalNumSteps >>= 8;
            ++index;
            unfilledNumBitsOfCurrentValue -= 8;
         }
         // Fill remaining bits (smaller than 8).
         if (unfilledNumBitsOfCurrentValue != 0) {
            charArray[index] = totalNumSteps << (8-unfilledNumBitsOfCurrentValue);
            bitsFilledInCurrentCharArrayUpToNow = unfilledNumBitsOfCurrentValue;
         }
      }
   }
}

void ROOT::Experimental::Detail::RColumnElement<double, ROOT::Experimental::EColumnType::kCustomDouble>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   double *doubleArray = reinterpret_cast<double *>(dst);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(src);
   if (kBitsOnStorage % 8 == 0) {
      const std::uint8_t numCharUsedToStoreOneValue = kBitsOnStorage / 8;
      for (std::size_t i = 0; i < count; ++i) {
         std::uint64_t totalNumSteps{0};
         for (int j = numCharUsedToStoreOneValue -1; j >= 0; --j) {
            totalNumSteps <<= 8;
            totalNumSteps += charArray[numCharUsedToStoreOneValue*i+j];
         }

         switch (totalNumSteps) {
            case 0:
               doubleArray[i] = std::numeric_limits<double>::infinity();
               break;
            case 1:
               doubleArray[i] = std::numeric_limits<double>::infinity() * (-1);
               break;
            case 2:
               doubleArray[i] = std::nan("");
               break;
            default:
               totalNumSteps -= 3; // because 0, 1 and 2 have special meanings, totalNumSteps = 3 represents the same value as fMin.
               doubleArray[i] = fMin + fStep * totalNumSteps;
               break;
         }
      }
   } else {
      int index = 0; // of charArray, count is index of doubleArray.
      short bitsPassedInCurrentCharArrayUpToNow = 0; // takes values from 0 to 7

      for (std::size_t i = 0; i < count; ++i) {
         short unfilledNumBitsOfCurrentValue = kBitsOnStorage;
         std::uint64_t totalNumSteps{0};
         // Phase 1: Use up unfinished char.
         if (bitsPassedInCurrentCharArrayUpToNow != 0) {
            short nBitsToRead = std::min(8 - bitsPassedInCurrentCharArrayUpToNow, (int)unfilledNumBitsOfCurrentValue);
            // e.g. get 3 bits with mask 0b 0000'0111
            short mask = (1 << nBitsToRead)-1;
            short numBitsToShiftMask = 8 - bitsPassedInCurrentCharArrayUpToNow - nBitsToRead;
            mask <<= numBitsToShiftMask;
            totalNumSteps += (charArray[index] & mask);
            totalNumSteps >>= numBitsToShiftMask;
            unfilledNumBitsOfCurrentValue -= nBitsToRead;
            bitsPassedInCurrentCharArrayUpToNow += nBitsToRead;
            R__ASSERT(bitsPassedInCurrentCharArrayUpToNow <= 8);
            if (bitsPassedInCurrentCharArrayUpToNow == 8) {
               ++index;
               bitsPassedInCurrentCharArrayUpToNow = 0;
            }
         }
         // Phase 2: Empty full char at a time.
         while (unfilledNumBitsOfCurrentValue > 8) {
            std::size_t filledNumBitsOfCurrentValue = kBitsOnStorage - unfilledNumBitsOfCurrentValue;
            totalNumSteps += (std::size_t)charArray[index] * ((std::size_t)1 << filledNumBitsOfCurrentValue);
            unfilledNumBitsOfCurrentValue -= 8;
            ++index;
         }
         // Phase 3: Only empty what you need.
         if (unfilledNumBitsOfCurrentValue != 0) {
            std::size_t toFill = charArray[index] >> (8-unfilledNumBitsOfCurrentValue);
            totalNumSteps += toFill * ((std::size_t)1 << (kBitsOnStorage - unfilledNumBitsOfCurrentValue));
            bitsPassedInCurrentCharArrayUpToNow += unfilledNumBitsOfCurrentValue;
         }
         switch (totalNumSteps) {
            case 0:
               doubleArray[i] = std::numeric_limits<double>::infinity();
               break;
            case 1:
               doubleArray[i] = std::numeric_limits<double>::infinity() * (-1);
               break;
            case 2:
               doubleArray[i] = std::nan("");
               break;
            default:
               totalNumSteps -= 3; // because 0, 1 and 2 have special meanings, totalNumSteps = 3 represents the same value as fMin.
               doubleArray[i] = fMin + fStep * totalNumSteps;
               break;
         }
      }
   }
}

void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kCustomFloat>::Pack(
  void *dst, void *src, std::size_t count) const
{
   float *floatArray = reinterpret_cast<float *>(src);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(dst);
   if (kBitsOnStorage % 8 == 0) { // use a faster procedure when %8 == 0
      // check here for inf, nan, overflow
      int numCharUsedToStoreOneValue = kBitsOnStorage / 8;
      for (std::size_t i = 0; i < count; ++i) {
         std::size_t totalNumSteps;
         if (std::isnan(floatArray[i])) totalNumSteps = 2; // NaN
         else if (floatArray[i] > fMax) totalNumSteps = 0; // overflow/positive infinity
         else if (floatArray[i] < fMin) totalNumSteps = 1; // underflow/negative infinity
         else totalNumSteps = std::llround( (floatArray[i]-fMin) / fStep) + 3;
         for (int j = 0; j < numCharUsedToStoreOneValue; ++j) {
            charArray[numCharUsedToStoreOneValue*i+j] = (totalNumSteps & 0xFF);
            totalNumSteps >>= 8;
         }
      }
   } else {
      int index = 0; // of charArray, count is index of doubleArray.
      short bitsFilledInCurrentCharArrayUpToNow = 0; // takes values from 0 to 7
      for (std::size_t i = 0; i < count; ++i) {
         short unfilledNumBitsOfCurrentValue = kBitsOnStorage;
         std::size_t totalNumSteps;
         if (std::isnan(floatArray[i])) totalNumSteps = 2; // NaN
         else if (floatArray[i] > fMax) totalNumSteps = 0; // overflow/positive infinity
         else if (floatArray[i] < fMin) totalNumSteps = 1; // underflow/negative infinity
         else totalNumSteps = std::llround( (floatArray[i]-fMin) / fStep) + 3;
         if (bitsFilledInCurrentCharArrayUpToNow != 0) {
            std::size_t nBitsToFill = std::min((int)kBitsOnStorage, (8 - bitsFilledInCurrentCharArrayUpToNow));
            short mask = (1 << nBitsToFill)-1;
            unsigned char ToAdd {static_cast<unsigned char>(totalNumSteps & mask)};
            ToAdd <<= (8 - bitsFilledInCurrentCharArrayUpToNow - nBitsToFill);
            charArray[index] += ToAdd; //>>= nBitsToFill);
            totalNumSteps >>= nBitsToFill;
            bitsFilledInCurrentCharArrayUpToNow += nBitsToFill;
            R__ASSERT(bitsFilledInCurrentCharArrayUpToNow <= 8);
            unfilledNumBitsOfCurrentValue -= nBitsToFill;
            if (bitsFilledInCurrentCharArrayUpToNow == 8) {
               ++index;
               bitsFilledInCurrentCharArrayUpToNow = 0;
            } else {
               continue;
            }
         }
         // Fill where entire 8 bits can be filled at once, with while loop instead of a for loop.
         while(unfilledNumBitsOfCurrentValue >= 8) {
            charArray[index] = (totalNumSteps & 0xFF);
            totalNumSteps >>= 8;
            ++index;
            unfilledNumBitsOfCurrentValue -= 8;
         }

         // Fill remaining bits (smaller than 8).
         if (unfilledNumBitsOfCurrentValue != 0) {
            charArray[index] = totalNumSteps << (8-unfilledNumBitsOfCurrentValue);
            bitsFilledInCurrentCharArrayUpToNow = unfilledNumBitsOfCurrentValue;
         }
      }
   }
}

void ROOT::Experimental::Detail::RColumnElement<float, ROOT::Experimental::EColumnType::kCustomFloat>::Unpack(
  void *dst, void *src, std::size_t count) const
{
   float *floatArray = reinterpret_cast<float *>(dst);
   unsigned char *charArray = reinterpret_cast<unsigned char *>(src);
   if (kBitsOnStorage % 8 == 0) {
      const std::uint8_t numCharUsedToStoreOneValue = kBitsOnStorage / 8;
      for (std::size_t i = 0; i < count; ++i) {
         std::uint64_t totalNumSteps{0};
         for (int j = numCharUsedToStoreOneValue -1; j >= 0; --j) {
            totalNumSteps <<= 8;
            totalNumSteps += charArray[numCharUsedToStoreOneValue*i+j];
         }

         switch (totalNumSteps) {
            case 0:
               floatArray[i] = std::numeric_limits<float>::infinity();
               break;
            case 1:
               floatArray[i] = std::numeric_limits<float>::infinity() * (-1);
               break;
            case 2:
               floatArray[i] = std::nan("");
               break;
            default:
               totalNumSteps -= 3; // because 0, 1 and 2 have special meanings, totalNumSteps = 3 represents the same value as fMin.
               floatArray[i] = fMin + fStep * totalNumSteps;
               break;
         } // end switch
      } // end for
   } else { // end if
      int index = 0; // of charArray, count is index of floatArray.
      short bitsPassedInCurrentCharArrayUpToNow = 0; // takes values from 0 to 7

      for (std::size_t i = 0; i < count; ++i) {
         short unfilledNumBitsOfCurrentValue = kBitsOnStorage;
         std::uint64_t totalNumSteps{0};
         // Phase 1: Use up unfinished char.
         if (bitsPassedInCurrentCharArrayUpToNow != 0) {
            short nBitsToRead = std::min(8 - bitsPassedInCurrentCharArrayUpToNow, (int)unfilledNumBitsOfCurrentValue);
            // e.g. get 3 bits with mask 0b 0000'0111
            short mask = (1 << nBitsToRead)-1;
            short numBitsToShiftMask = 8 - bitsPassedInCurrentCharArrayUpToNow - nBitsToRead;
            mask <<= numBitsToShiftMask;
            totalNumSteps += (charArray[index] & mask);
            totalNumSteps >>= numBitsToShiftMask;
            unfilledNumBitsOfCurrentValue -= nBitsToRead;
            bitsPassedInCurrentCharArrayUpToNow += nBitsToRead;
            R__ASSERT(bitsPassedInCurrentCharArrayUpToNow <= 8);
            if (bitsPassedInCurrentCharArrayUpToNow == 8) {
               ++index;
               bitsPassedInCurrentCharArrayUpToNow = 0;
            }
         }
         // Phase 2: Empty full char at a time.
         while (unfilledNumBitsOfCurrentValue > 8) {
            std::size_t filledNumBitsOfCurrentValue = kBitsOnStorage - unfilledNumBitsOfCurrentValue;
            totalNumSteps += (std::size_t)charArray[index] * ((std::size_t)1 << filledNumBitsOfCurrentValue);
            unfilledNumBitsOfCurrentValue -= 8;
            ++index;
         }
         // Phase 3: Only empty what you need.
         if (unfilledNumBitsOfCurrentValue != 0) {
            std::size_t toFill = charArray[index] >> (8-unfilledNumBitsOfCurrentValue);
            totalNumSteps += toFill * ((std::size_t)1 << (kBitsOnStorage - unfilledNumBitsOfCurrentValue));
            bitsPassedInCurrentCharArrayUpToNow += unfilledNumBitsOfCurrentValue;
         }
         switch (totalNumSteps) {
            case 0:
               floatArray[i] = std::numeric_limits<float>::infinity();
               break;
            case 1:
               floatArray[i] = std::numeric_limits<float>::infinity() * (-1);
               break;
            case 2:
               floatArray[i] = std::nan("");
               break;
            default:
               totalNumSteps -= 3; // because 0, 1 and 2 have special meanings, totalNumSteps = 3 represents the same value as fMin.
               floatArray[i] = fMin + fStep * totalNumSteps;
               break;
         }
      }
   }
}
