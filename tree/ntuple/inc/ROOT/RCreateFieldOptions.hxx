/// \file ROOT/RCreateFieldOptions.hxx
/// \ingroup NTuple
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2024-12-17

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCreateFieldOptions
#define ROOT_RCreateFieldOptions

namespace ROOT {

struct RCreateFieldOptions {
private:
   /// If true, failing to create a field will return a RInvalidField instead of throwing an exception.
   bool fReturnInvalidOnError = false;
   /// If true, fields with a user defined type that have no available dictionaries will be reconstructed
   /// as record fields from the on-disk information; otherwise, they will cause an error.
   bool fEmulateUnknownTypes = false;

public:
   void SetReturnInvalidOnError(bool v) { fReturnInvalidOnError = v; }
   bool GetReturnInvalidOnError() const { return fReturnInvalidOnError; }

   void SetEmulateUnknownTypes(bool v) { fEmulateUnknownTypes = v; }
   bool GetEmulateUnknownTypes() const { return fEmulateUnknownTypes; }
};

} // namespace ROOT

#endif
