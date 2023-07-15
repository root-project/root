/// \file REntry.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>

#include <algorithm>

void ROOT::Experimental::REntry::AddValue(Detail::RFieldBase::RValue &&value)
{
   fValues.emplace_back(std::move(value));
}

void ROOT::Experimental::REntry::CaptureValueUnsafe(std::string_view fieldName, void *where)
{
   for (std::size_t i = 0; i < fValues.size(); ++i) {
      if (fValues[i].GetField()->GetName() != fieldName)
         continue;
      // TODO: remove const_cast
      fValues[i] =
         Detail::RFieldBase::RValue(const_cast<Detail::RFieldBase *>(fValues[i].GetField())->CaptureValue(where));
      return;
   }
   throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
}
