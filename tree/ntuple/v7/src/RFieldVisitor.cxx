/// \file RFieldVisitor.cxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.leisibach@gmail.com>
/// \date 2019-06-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


//----------------------------- RPrepareVisitor --------------------------------

void ROOT::Experimental::RPrepareVisitor::VisitField(const ROOT::RFieldBase &field)
{
   auto subfields = field.GetConstSubfields();
   for (auto f : subfields) {
      RPrepareVisitor visitor;
      f->AcceptVisitor(visitor);
      fNumFields += visitor.fNumFields;
      fDeepestLevel = std::max(fDeepestLevel, 1 + visitor.fDeepestLevel);
   }
}

void ROOT::Experimental::RPrepareVisitor::VisitFieldZero(const ROOT::RFieldZero &field)
{
   VisitField(field);
   fNumFields--;
   fDeepestLevel--;
}


//---------------------------- RPrintSchemaVisitor -----------------------------


void ROOT::Experimental::RPrintSchemaVisitor::SetDeepestLevel(int d)
{
   fDeepestLevel = d;
}

void ROOT::Experimental::RPrintSchemaVisitor::SetNumFields(int n)
{
   fNumFields = n;
   SetAvailableSpaceForStrings();
}

void ROOT::Experimental::RPrintSchemaVisitor::VisitField(const ROOT::RFieldBase &field)
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

void ROOT::Experimental::RPrintSchemaVisitor::VisitFieldZero(const ROOT::RFieldZero &fieldZero)
{
   auto fieldNo = 1;
   for (auto f : fieldZero.GetConstSubfields()) {
      RPrintSchemaVisitor visitor(*this);
      visitor.fFieldNo = fieldNo++;
      f->AcceptVisitor(visitor);
   }
}


//--------------------------- RPrintValueVisitor -------------------------------

void ROOT::Experimental::RPrintValueVisitor::PrintIndent()
{
   if (fPrintOptions.fPrintSingleLine)
      return;

   for (unsigned int i = 0; i < fLevel; ++i)
      fOutput << "  ";
}

void ROOT::Experimental::RPrintValueVisitor::PrintName(const ROOT::RFieldBase &field)
{
   if (fPrintOptions.fPrintName)
      fOutput << "\"" << field.GetFieldName() << "\": ";
}

void ROOT::Experimental::RPrintValueVisitor::PrintCollection(const ROOT::RFieldBase &field)
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

void ROOT::Experimental::RPrintValueVisitor::PrintRecord(const ROOT::RFieldBase &field)
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

void ROOT::Experimental::RPrintValueVisitor::VisitField(const ROOT::RFieldBase &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "\"<unsupported type: " << field.GetTypeName() << ">\"";
}

void ROOT::Experimental::RPrintValueVisitor::VisitBoolField(const ROOT::RField<bool> &field)
{
   PrintIndent();
   PrintName(field);
   if (fValue.GetRef<bool>())
      fOutput << "true";
   else
      fOutput << "false";
}

void ROOT::Experimental::RPrintValueVisitor::VisitDoubleField(const ROOT::RField<double> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<double>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitFloatField(const ROOT::RField<float> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<float>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitByteField(const ROOT::RField<std::byte> &field)
{
   PrintIndent();
   PrintName(field);
   char prev = std::cout.fill();
   fOutput << "0x" << std::setw(2) << std::setfill('0') << std::hex << (fValue.GetRef<unsigned char>() & 0xff);
   fOutput << std::resetiosflags(std::ios_base::basefield);
   std::cout.fill(prev);
}

void ROOT::Experimental::RPrintValueVisitor::VisitCharField(const ROOT::RField<char> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<char>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt8Field(const ROOT::RIntegralField<std::int8_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << static_cast<int>(fValue.GetRef<std::int8_t>());
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt16Field(const ROOT::RIntegralField<std::int16_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::int16_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt32Field(const ROOT::RIntegralField<std::int32_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::int32_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt64Field(const ROOT::RIntegralField<std::int64_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::int64_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitStringField(const ROOT::RField<std::string> &field)
{
   PrintIndent();
   PrintName(field);
   // TODO(jblomer): escape double quotes
   fOutput << "\"" << fValue.GetRef<std::string>() << "\"";
}

void ROOT::Experimental::RPrintValueVisitor::VisitUInt8Field(const ROOT::RIntegralField<std::uint8_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << static_cast<int>(fValue.GetRef<std::uint8_t>());
}

void ROOT::Experimental::RPrintValueVisitor::VisitUInt16Field(const ROOT::RIntegralField<std::uint16_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::uint16_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitUInt32Field(const ROOT::RIntegralField<std::uint32_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::uint32_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitUInt64Field(const ROOT::RIntegralField<std::uint64_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << fValue.GetRef<std::uint64_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitCardinalityField(const ROOT::RCardinalityField &field)
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

void ROOT::Experimental::RPrintValueVisitor::VisitBitsetField(const ROOT::RBitsetField &field)
{
   constexpr auto nBitsULong = sizeof(unsigned long) * 8;
   const auto *asULongArray = fValue.GetPtr<unsigned long>().get();

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

void ROOT::Experimental::RPrintValueVisitor::VisitArrayField(const ROOT::RArrayField &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitArrayAsRVecField(const ROOT::RArrayAsRVecField &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitStreamerField(const ROOT::RStreamerField &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "<streamer mode>";
}

void ROOT::Experimental::RPrintValueVisitor::VisitClassField(const ROOT::RClassField &field)
{
   PrintRecord(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitTObjectField(const ROOT::RField<TObject> &field)
{
   PrintRecord(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitRecordField(const ROOT::RRecordField &field)
{
   PrintRecord(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitNullableField(const ROOT::RNullableField &field)
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

void ROOT::Experimental::RPrintValueVisitor::VisitEnumField(const ROOT::REnumField &field)
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

void ROOT::Experimental::RPrintValueVisitor::VisitAtomicField(const ROOT::RAtomicField &field)
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

void ROOT::Experimental::RPrintValueVisitor::VisitProxiedCollectionField(const ROOT::RProxiedCollectionField &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitVectorField(const ROOT::RVectorField &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitVectorBoolField(const ROOT::RField<std::vector<bool>> &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitRVecField(const ROOT::RRVecField &field)
{
   PrintCollection(field);
}

//---------------------------- RNTupleFormatter --------------------------------


std::string ROOT::Experimental::RNTupleFormatter::FitString(const std::string &str, int availableSpace)
{
   int strSize{static_cast<int>(str.size())};
   if (strSize <= availableSpace)
      return str + std::string(availableSpace - strSize, ' ');
   else if (availableSpace < 3)
      return std::string(availableSpace, '.');
   return std::string(str, 0, availableSpace - 3) + "...";
}
