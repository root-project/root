/// \file RROOT/RHashValueVisitor.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFieldVisitor.hxx>

namespace ROOT {
namespace Experimental {
namespace Internal {
// clang-format off
/**
\class ROOT::Experimental::Internal::RHashValueVisitor
\ingroup NTuple
\brief Provides the possibility to get the hash (using `std::hash`) for values of a subset of field types.

The supported field types are all arithmetic field types, `bool`, `char` and `std::string` fields. An exception will be
thrown when trying to get a hash for any other field type.
*/
// clang-format on
class RHashValueVisitor : public Detail::RFieldVisitor {
private:
   /// A pointer to the value to hash.
   void *fValuePtr;
   /// The resulting hash.
   std::size_t fHash;

public:
   RHashValueVisitor(void *valuePtr) : fValuePtr(valuePtr) {}

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the resulting hash.
   ///
   /// \return The resulting hash.
   std::size_t GetHash() const { return fHash; }

   void VisitField(const RFieldBase &field) final;

   void VisitBoolField(const RField<bool> &field) final;
   void VisitDoubleField(const RField<double> &field) final;
   void VisitFloatField(const RField<float> &field) final;
   void VisitCharField(const RField<char> &field) final;
   void VisitStringField(const RField<std::string> &field) final;
   void VisitInt8Field(const RField<std::int8_t> &field) final;
   void VisitInt16Field(const RField<std::int16_t> &field) final;
   void VisitInt32Field(const RField<std::int32_t> &field) final;
   void VisitInt64Field(const RField<std::int64_t> &field) final;
   void VisitUInt8Field(const RField<std::uint8_t> &field) final;
   void VisitUInt16Field(const RField<std::uint16_t> &field) final;
   void VisitUInt32Field(const RField<std::uint32_t> &field) final;
   void VisitUInt64Field(const RField<std::uint64_t> &field) final;
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT
