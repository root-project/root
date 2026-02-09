/// \file RNTupleTypes.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2026-02-00

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <TError.h>

#include <string>

/// See GetReserved(): we ignore the reserved flag since we don't use it anywhere currently.
void ROOT::RNTupleLocator::SetReserved(std::uint8_t reserved)
{
   if (reserved > 1) {
      throw RException(R__FAIL("Overflow of the locator reserved field"));
   }

   fFlagsAndNBytes = (fFlagsAndNBytes & ~kMaskReservedBit) | (std::uint64_t(reserved) << 60);
}

void ROOT::RNTupleLocator::SetNBytesOnStorage(std::uint64_t nBytesOnStorage)
{
   if (nBytesOnStorage & kMaskFlags) {
      throw RException(R__FAIL("overflow in on-disk size of locator: " + std::to_string(nBytesOnStorage)));
   }

   fFlagsAndNBytes = (fFlagsAndNBytes & kMaskFlags) | nBytesOnStorage;
}

ROOT::RNTupleLocator::ELocatorType ROOT::RNTupleLocator::GetType() const
{
   std::uint64_t compactType = fFlagsAndNBytes >> 61;
   switch (compactType) {
   case 0: return kTypeFile;
   case 1: return kTypeDAOS;
   case 2: return kTypePageZero;
   case 3: return kTypeUnknown;
   case 4: return Internal::kTestLocatorType;
   default: break;
   }
   R__ASSERT(false);
   return kTypeUnknown;
}

void ROOT::RNTupleLocator::SetType(ELocatorType type)
{
   std::uint64_t compactType;
   switch (type) {
   case kTypeFile: compactType = 0; break;
   case kTypeDAOS: compactType = 1; break;
   case kTypePageZero: compactType = 2; break;
   case kTypeUnknown: compactType = 3; break;
   default:
      if (type == Internal::kTestLocatorType)
         compactType = 4;
      else
         throw RException(R__FAIL("invalid locator type: " + std::to_string(type)));
   }

   fFlagsAndNBytes = (fFlagsAndNBytes & ~kMaskType) | (compactType << 61);
}

void ROOT::RNTupleLocator::SetPosition(std::uint64_t position)
{
   if (GetType() != kTypeFile)
      throw RException(R__FAIL("cannot set position as 64bit offset for type " + std::to_string(GetType())));
   fPosition = position;
}

void ROOT::RNTupleLocator::SetPosition(RNTupleLocatorObject64 position)
{
   if (GetType() != kTypeDAOS)
      throw RException(R__FAIL("cannot set position as 64bit object for type " + std::to_string(GetType())));
   fPosition = position.GetLocation();
}

std::uint64_t ROOT::Internal::RNTupleLocatorHelper<std::uint64_t>::Get(const RNTupleLocator &loc)
{
   if (loc.GetType() != ROOT::RNTupleLocator::kTypeFile)
      throw RException(R__FAIL("cannot retrieve position as 64bit offset for type " + std::to_string(loc.GetType())));
   return loc.fPosition;
}

ROOT::RNTupleLocatorObject64
ROOT::Internal::RNTupleLocatorHelper<ROOT::RNTupleLocatorObject64>::Get(const RNTupleLocator &loc)
{
   if (loc.GetType() != ROOT::RNTupleLocator::kTypeDAOS)
      throw RException(R__FAIL("cannot retrieve position as 64bit object for type " + std::to_string(loc.GetType())));
   return RNTupleLocatorObject64{loc.fPosition};
}
