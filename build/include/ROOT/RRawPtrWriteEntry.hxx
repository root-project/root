/// \file ROOT/RRawPtrWriteEntry.hxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2025-03-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRawPtrWriteEntry
#define ROOT_RRawPtrWriteEntry

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RFieldToken.hxx>
#include <ROOT/RError.hxx>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ROOT {

class RNTupleModel;

namespace Experimental {

class RNTupleFillContext;

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RRawPtrWriteEntry
\ingroup NTuple
\brief A container of const raw pointers, corresponding to a row in the data set

This class can be used to write constant data products in frameworks.  All other users are encouraged to use the API
provided by REntry, with safe interfaces, type checks, and shared object ownership.
*/
// clang-format on
class RRawPtrWriteEntry {
   friend class ROOT::RNTupleModel;
   friend class ROOT::Experimental::RNTupleFillContext;

private:
   /// The entry must be linked to a specific model, identified by a model ID
   std::uint64_t fModelId = 0;
   /// The entry and its tokens are also linked to a specific schema, identified by a schema ID
   std::uint64_t fSchemaId = 0;
   /// Corresponds to the fields of the linked model
   std::vector<ROOT::RFieldBase *> fFields;
   /// The raw pointers corresponding to the fields
   std::vector<const void *> fRawPtrs;
   /// For fast lookup of token IDs given a (sub)field name present in the entry
   std::unordered_map<std::string, std::size_t> fFieldName2Token;

   explicit RRawPtrWriteEntry(std::uint64_t modelId, std::uint64_t schemaId) : fModelId(modelId), fSchemaId(schemaId) {}

   void AddField(ROOT::RFieldBase &field)
   {
      fFieldName2Token[field.GetQualifiedFieldName()] = fFields.size();
      fFields.emplace_back(&field);
      fRawPtrs.emplace_back(nullptr);
   }

   std::size_t Append()
   {
      std::size_t bytesWritten = 0;
      for (std::size_t i = 0; i < fFields.size(); i++) {
         bytesWritten += fFields[i]->Append(fRawPtrs[i]);
      }
      return bytesWritten;
   }

   void EnsureMatchingModel(RFieldToken token) const
   {
      if (fSchemaId != token.fSchemaId) {
         throw RException(R__FAIL("invalid token for this entry, "
                                  "make sure to use a token from a model with the same schema as this entry."));
      }
   }

public:
   RRawPtrWriteEntry(const RRawPtrWriteEntry &other) = delete;
   RRawPtrWriteEntry &operator=(const RRawPtrWriteEntry &other) = delete;
   RRawPtrWriteEntry(RRawPtrWriteEntry &&other) = default;
   RRawPtrWriteEntry &operator=(RRawPtrWriteEntry &&other) = default;
   ~RRawPtrWriteEntry() = default;

   /// The ordinal of the (sub)field fieldName; can be used in other methods to address the corresponding value
   RFieldToken GetToken(std::string_view fieldName) const
   {
      auto it = fFieldName2Token.find(std::string(fieldName));
      if (it == fFieldName2Token.end()) {
         throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
      }
      return RFieldToken(it->second, fSchemaId);
   }

   template <typename T>
   void BindRawPtr(RFieldToken token, const T *rawPtr)
   {
      EnsureMatchingModel(token);
      fRawPtrs[token.fIndex] = rawPtr;
   }

   template <typename T>
   void BindRawPtr(std::string_view fieldName, const T *rawPtr)
   {
      BindRawPtr(GetToken(fieldName), rawPtr);
   }

   std::uint64_t GetModelId() const { return fModelId; }
   std::uint64_t GetSchemaId() const { return fSchemaId; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
