/// \file ROOT/RFieldToken.hxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2025-03-19

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RFieldToken
#define ROOT_RFieldToken

#include <cstddef> // for std::size_t
#include <cstdint> // for std::uint64_t

namespace ROOT {

class REntry;
class RNTupleModel;

namespace Experimental {
namespace Detail {
class RRawPtrWriteEntry;
} // namespace Detail
} // namespace Experimental

// clang-format off
/**
\class ROOT::RFieldToken
\ingroup NTuple
\brief A field token identifies a (sub)field in an entry

It can be used for fast indexing in REntry's methods, e.g. REntry::BindValue(). The field token can also be created by the model.
*/
// clang-format on
class RFieldToken {
   friend class REntry;
   friend class RNTupleModel;
   friend class Experimental::Detail::RRawPtrWriteEntry;

   std::size_t fIndex = 0;                      ///< The index of the field (top-level or registered subfield)
   std::uint64_t fSchemaId = std::uint64_t(-1); ///< Safety check to prevent tokens from other models being used
   RFieldToken(std::size_t index, std::uint64_t schemaId) : fIndex(index), fSchemaId(schemaId) {}

public:
   RFieldToken() = default; // The default constructed token cannot be used by any entry
};

} // namespace ROOT

#endif
