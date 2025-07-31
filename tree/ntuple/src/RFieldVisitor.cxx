/// \file RFieldVisitor.cxx
/// \ingroup NTuple
/// \author Simon Leisibach <simon.leisibach@gmail.com>
/// \date 2019-06-11

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleView.hxx>

#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


//----------------------------- RPrepareVisitor --------------------------------

void ROOT::Internal::RPrepareVisitor::VisitField(const ROOT::RFieldBase &field)
{
   auto subfields = field.GetConstSubfields();
   for (auto f : subfields) {
      RPrepareVisitor visitor;
      f->AcceptVisitor(visitor);
      fNumFields += visitor.fNumFields;
      fDeepestLevel = std::max(fDeepestLevel, 1 + visitor.fDeepestLevel);
   }
}

void ROOT::Internal::RPrepareVisitor::VisitFieldZero(const ROOT::RFieldZero &field)
{
   VisitField(field);
   fNumFields--;
   fDeepestLevel--;
}


//---------------------------- RPrintSchemaVisitor -----------------------------

void ROOT::Internal::RPrintSchemaVisitor::SetDeepestLevel(int d)
{
   fDeepestLevel = d;
}

void ROOT::Internal::RPrintSchemaVisitor::SetNumFields(int n)
{
   fNumFields = n;
   SetAvailableSpaceForStrings();
}

void ROOT::Internal::RPrintSchemaVisitor::VisitField(const ROOT::RFieldBase &field)
{
   fOutput << fFrameSymbol << ' ';

   std::string key = fTreePrefix;
   key += "Field " + fFieldNoPrefix + std::to_string(fFieldNo);
   fOutput << RNTupleFormatter::FitString(key, fAvailableSpaceKeyString);
   fOutput << " : ";

   std::string value = field.GetFieldName();
   if (!field.GetTypeName().empty())
      value += " (" + field.GetTypeName() + ")";
   fOutput << RNTupleFormatter::FitString(value, fAvailableSpaceValueString);
   fOutput << fFrameSymbol << std::endl;

   auto subfields = field.GetConstSubfields();
   auto fieldNo = 1;
   for (auto iField = subfields.begin(); iField != subfields.end();) {
      RPrintSchemaVisitor visitor(*this);
      visitor.fFieldNo = fieldNo++;
      visitor.fFieldNoPrefix += std::to_string(fFieldNo) + ".";

      auto f = *iField;
      ++iField;
      // TODO(jblomer): implement tree drawing
      visitor.fTreePrefix += "  ";
      f->AcceptVisitor(visitor);
   }
}

void ROOT::Internal::RPrintSchemaVisitor::VisitFieldZero(const ROOT::RFieldZero &fieldZero)
{
   auto fieldNo = 1;
   for (auto f : fieldZero.GetConstSubfields()) {
      RPrintSchemaVisitor visitor(*this);
      visitor.fFieldNo = fieldNo++;
      f->AcceptVisitor(visitor);
   }
}


//--------------------------- RPrintValueVisitor -------------------------------

void ROOT::Internal::RPrintValueVisitor::PrintIndent()
{
   if (fPrintOptions.fPrintSingleLine)
      return;

   for (unsigned int i = 0; i < fLevel; ++i)
      fOutput << "  ";
}

void ROOT::Internal::RPrintValueVisitor::PrintName(const ROOT::RFieldBase &field)
{
   if (fPrintOptions.fPrintName)
      fOutput << "\"" << field.GetFieldName() << "\": ";
}

void ROOT::Internal::RPrintValueVisitor::PrintCollection(const ROOT::RFieldBase &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "[";
   auto elems = field.SplitValue(fValue);
   for (auto iValue = elems.begin(); iValue != elems.end(); ) {
      RPrintOptions options;
      options.fPrintSingleLine = true;
      options.fPrintName = false;
      RPrintValueVisitor elemVisitor(*iValue, fOutput, 0 /* level */, options);
      iValue->GetField().AcceptVisitor(elemVisitor);

      if (++iValue == elems.end())
         break;
      else
         fOutput << ", ";
   }
   fOutput << "]";
}

void ROOT::Internal::RPrintValueVisitor::PrintRecord(const ROOT::RFieldBase &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "{";
   auto elems = field.SplitValue(fValue);
   for (auto iValue = elems.begin(); iValue != elems.end();) {
      if (!fPrintOptions.fPrintSingleLine)
         fOutput << std::endl;

      RPrintOptions options;
      options.fPrintSingleLine = fPrintOptions.fPrintSingleLine;
      RPrintValueVisitor visitor(*iValue, fOutput, fLevel + 1, options);
      iValue->GetField().AcceptVisitor(visitor);

      if (++iValue == elems.end()) {
         if (!fPrintOptions.fPrintSingleLine)
            fOutput << std::endl;
         break;
      } else {
         fOutput << ",";
         if (fPrintOptions.fPrintSingleLine)
            fOutput << " ";
      }
   }
   PrintIndent();
   fOutput << "}";
}

void ROOT::Internal::RPrintValueVisitor::VisitField(const ROOT::RFieldBase &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "\"<unsupported type: " << field.GetTypeName() << ">\"";
}

void ROOT::Internal::RPrintValueVisitor::VisitBoolField(const ROOT::RField<bool> &field)
{
   PrintIndent();
   PrintName(field);
   if (fValue.GetRef<bool>())
      fOutput << "true";
   else
      fOutput << "false";
}

void ROOT::Internal::RPrintValueVisitor::VisitDoubleField(const ROOT::RField<double> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<double>();
}

void ROOT::Internal::RPrintValueVisitor::VisitFloatField(const ROOT::RField<float> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<float>();
}

void ROOT::Internal::RPrintValueVisitor::VisitByteField(const ROOT::RField<std::byte> &field)
{
   PrintIndent();
   PrintName(field);
   char prev = std::cout.fill();
   auto value = std::to_integer<unsigned int>(fValue.GetRef<std::byte>());
   fOutput << "0x" << std::setw(2) << std::setfill('0') << std::hex << value;
   fOutput << std::resetiosflags(std::ios_base::basefield);
   std::cout.fill(prev);
}

void ROOT::Internal::RPrintValueVisitor::VisitCharField(const ROOT::RField<char> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<char>();
}

void ROOT::Internal::RPrintValueVisitor::VisitInt8Field(const ROOT::RIntegralField<std::int8_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << static_cast<int>(fValue.GetRef<std::int8_t>());
}

void ROOT::Internal::RPrintValueVisitor::VisitInt16Field(const ROOT::RIntegralField<std::int16_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::int16_t>();
}

void ROOT::Internal::RPrintValueVisitor::VisitInt32Field(const ROOT::RIntegralField<std::int32_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::int32_t>();
}

void ROOT::Internal::RPrintValueVisitor::VisitInt64Field(const ROOT::RIntegralField<std::int64_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::int64_t>();
}

void ROOT::Internal::RPrintValueVisitor::VisitStringField(const ROOT::RField<std::string> &field)
{
   PrintIndent();
   PrintName(field);
   // TODO(jblomer): escape double quotes
   fOutput << "\"" << fValue.GetRef<std::string>() << "\"";
}

void ROOT::Internal::RPrintValueVisitor::VisitUInt8Field(const ROOT::RIntegralField<std::uint8_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << static_cast<int>(fValue.GetRef<std::uint8_t>());
}

void ROOT::Internal::RPrintValueVisitor::VisitUInt16Field(const ROOT::RIntegralField<std::uint16_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::uint16_t>();
}

void ROOT::Internal::RPrintValueVisitor::VisitUInt32Field(const ROOT::RIntegralField<std::uint32_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::uint32_t>();
}

void ROOT::Internal::RPrintValueVisitor::VisitUInt64Field(const ROOT::RIntegralField<std::uint64_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::uint64_t>();
}

void ROOT::Internal::RPrintValueVisitor::VisitCardinalityField(const ROOT::RCardinalityField &field)
{
   PrintIndent();
   PrintName(field);
   if (field.As32Bit()) {
      fOutput << fValue.GetRef<std::uint32_t>();
      return;
   }
   if (field.As64Bit()) {
      fOutput << fValue.GetRef<std::uint64_t>();
      return;
   }
   R__ASSERT(false && "unsupported cardinality size type");
}

void ROOT::Internal::RPrintValueVisitor::VisitBitsetField(const ROOT::RBitsetField &field)
{
   constexpr auto nBitsULong = sizeof(unsigned long) * 8;
   const auto *asULongArray = static_cast<unsigned long *>(fValue.GetPtr<void>().get());

   PrintIndent();
   PrintName(field);
   fOutput << "\"";
   std::size_t i = 0;
   std::string str;
   for (std::size_t word = 0; word < (field.GetN() + nBitsULong - 1) / nBitsULong; ++word) {
      for (std::size_t mask = 0; (mask < nBitsULong) && (i < field.GetN()); ++mask, ++i) {
         bool isSet = (asULongArray[word] & (static_cast<unsigned long>(1) << mask)) != 0;
         str = std::to_string(isSet) + str;
      }
   }
   fOutput << str << "\"";
}

void ROOT::Internal::RPrintValueVisitor::VisitArrayField(const ROOT::RArrayField &field)
{
   PrintCollection(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitArrayAsRVecField(const ROOT::RArrayAsRVecField &field)
{
   PrintCollection(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitStreamerField(const ROOT::RStreamerField &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "<streamer mode>";
}

void ROOT::Internal::RPrintValueVisitor::VisitClassField(const ROOT::RClassField &field)
{
   PrintRecord(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitTObjectField(const ROOT::RField<TObject> &field)
{
   PrintRecord(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitRecordField(const ROOT::RRecordField &field)
{
   PrintRecord(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitNullableField(const ROOT::RNullableField &field)
{
   PrintIndent();
   PrintName(field);
   auto elems = field.SplitValue(fValue);
   if (elems.empty()) {
      fOutput << "null";
   } else {
      RPrintOptions options;
      options.fPrintSingleLine = true;
      options.fPrintName = false;
      RPrintValueVisitor visitor(elems[0], fOutput, fLevel, options);
      elems[0].GetField().AcceptVisitor(visitor);
   }
}

void ROOT::Internal::RPrintValueVisitor::VisitEnumField(const ROOT::REnumField &field)
{
   PrintIndent();
   PrintName(field);
   auto intValue = field.SplitValue(fValue)[0];
   RPrintOptions options;
   options.fPrintSingleLine = true;
   options.fPrintName = false;
   RPrintValueVisitor visitor(intValue, fOutput, fLevel, options);
   intValue.GetField().AcceptVisitor(visitor);
}

void ROOT::Internal::RPrintValueVisitor::VisitAtomicField(const ROOT::RAtomicField &field)
{
   PrintIndent();
   PrintName(field);
   auto itemValue = field.SplitValue(fValue)[0];
   RPrintOptions options;
   options.fPrintSingleLine = true;
   options.fPrintName = false;
   RPrintValueVisitor visitor(itemValue, fOutput, fLevel, options);
   itemValue.GetField().AcceptVisitor(visitor);
}

void ROOT::Internal::RPrintValueVisitor::VisitProxiedCollectionField(const ROOT::RProxiedCollectionField &field)
{
   PrintCollection(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitVectorField(const ROOT::RVectorField &field)
{
   PrintCollection(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitVectorBoolField(const ROOT::RField<std::vector<bool>> &field)
{
   PrintCollection(field);
}

void ROOT::Internal::RPrintValueVisitor::VisitRVecField(const ROOT::RRVecField &field)
{
   PrintCollection(field);
}

//---------------------------- RNTupleFormatter --------------------------------

std::string ROOT::Internal::RNTupleFormatter::FitString(const std::string &str, int availableSpace)
{
   int strSize{static_cast<int>(str.size())};
   if (strSize <= availableSpace)
      return str + std::string(availableSpace - strSize, ' ');
   else if (availableSpace < 3)
      return std::string(availableSpace, '.');
   return std::string(str, 0, availableSpace - 3) + "...";
}
