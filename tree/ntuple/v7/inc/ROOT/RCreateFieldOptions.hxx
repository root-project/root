/// \file ROOT/RCreateFieldOptions.hxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2024-12-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RCreateFieldOptions
#define ROOT7_RCreateFieldOptions

namespace ROOT::Experimental {

struct RCreateFieldOptions {
   /// If true, failing to create a field will return a RInvalidField instead of throwing an exception.
   bool fReturnInvalidOnError = false;
   /// If true, fields with a user defined type that have no available dictionaries will be reconstructed
   /// as record fields from the on-disk information; otherwise, they will cause an error.
   bool fEmulateUnknownTypes = false;
};

} // namespace ROOT::Experimental

#endif
