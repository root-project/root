/// \file RHashValueVisitor.cxx
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

#include <ROOT/RHashValueVisitor.hxx>

void ROOT::Experimental::Internal::RHashValueVisitor::VisitField(const RFieldBase &field)
{
   throw RException(R__FAIL("hashing is not supported for fields of type " + field.GetTypeName()));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitBoolField(const RField<bool> & /* field */)
{
   fHash = std::hash<bool>()(*static_cast<bool *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitDoubleField(const RField<double> & /* field */)
{
   fHash = std::hash<double>()(*static_cast<double *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitFloatField(const RField<float> & /* field */)
{
   fHash = std::hash<float>()(*static_cast<float *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitCharField(const RField<char> & /* field */)
{
   fHash = std::hash<char>()(*static_cast<char *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitStringField(const RField<std::string> & /* field */)
{
   fHash = std::hash<std::string>()(*static_cast<std::string *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitInt8Field(const RField<std::int8_t> & /* field */)
{
   fHash = std::hash<std::int8_t>()(*static_cast<std::int8_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitInt16Field(const RField<std::int16_t> & /* field */)
{
   fHash = std::hash<std::int16_t>()(*static_cast<std::int16_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitInt32Field(const RField<std::int32_t> & /* field */)
{
   fHash = std::hash<std::int32_t>()(*static_cast<std::int32_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitInt64Field(const RField<std::int64_t> & /* field */)
{
   fHash = std::hash<std::int64_t>()(*static_cast<std::int64_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitUInt8Field(const RField<std::uint8_t> & /* field */)
{
   fHash = std::hash<std::uint8_t>()(*static_cast<std::uint8_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitUInt16Field(const RField<std::uint16_t> & /* field */)
{
   fHash = std::hash<std::uint16_t>()(*static_cast<std::uint16_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitUInt32Field(const RField<std::uint32_t> & /* field */)
{
   fHash = std::hash<std::uint32_t>()(*static_cast<std::uint32_t *>(fValuePtr));
}

void ROOT::Experimental::Internal::RHashValueVisitor::VisitUInt64Field(const RField<std::uint64_t> & /* field */)
{
   fHash = std::hash<std::uint64_t>()(*static_cast<std::uint64_t *>(fValuePtr));
}
