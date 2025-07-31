/// \file ROOT/RFieldVisitor.hxx
/// \ingroup NTuple
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-06-11

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RFieldVisitor
#define ROOT_RFieldVisitor

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ROOT {
namespace Detail {

// clang-format off
/**
\class ROOT::Detail::RFieldVisitor
\ingroup NTuple
\brief Abstract base class for classes implementing the visitor design pattern.

RFieldVisitor::VisitField() is invoked by RFieldBase::AcceptVisitor().

**Example: creating a custom field visitor**
~~~ {.cpp}
// 1. Define your visitor:
class MyVisitor : public RFieldVisitor {
public:
   // This is the only method you need to define. The others will default to calling this.
   // Only implement other methods if you need special logic for a specific field type.
   void VisitField(const ROOT::RFieldBase &field) final {
      // ... do custom logic here ...
   }
};

// 2. Use it:
const auto &field = reader->GetModel().GetConstFieldZero();
MyVisitor visitor;
visitor.VisitField(field);
~~~

As an example of a concrete use case, see Internal::RPrintSchemaVisitor.
*/
// clang-format on
class RFieldVisitor {
public:
   virtual void VisitField(const ROOT::RFieldBase &field) = 0;
   virtual void VisitFieldZero(const ROOT::RFieldZero &field) { VisitField(field); }
   virtual void VisitArrayField(const ROOT::RArrayField &field) { VisitField(field); }
   virtual void VisitArrayAsRVecField(const ROOT::RArrayAsRVecField &field) { VisitField(field); }
   virtual void VisitAtomicField(const ROOT::RAtomicField &field) { VisitField(field); }
   virtual void VisitBitsetField(const ROOT::RBitsetField &field) { VisitField(field); }
   virtual void VisitBoolField(const ROOT::RField<bool> &field) { VisitField(field); }
   virtual void VisitClassField(const ROOT::RClassField &field) { VisitField(field); }
   virtual void VisitTObjectField(const ROOT::RField<TObject> &field) { VisitField(field); }
   virtual void VisitStreamerField(const ROOT::RStreamerField &field) { VisitField(field); }
   virtual void VisitProxiedCollectionField(const ROOT::RProxiedCollectionField &field) { VisitField(field); }
   virtual void VisitRecordField(const ROOT::RRecordField &field) { VisitField(field); }
   virtual void VisitCardinalityField(const ROOT::RCardinalityField &field) { VisitField(field); }
   virtual void VisitDoubleField(const ROOT::RField<double> &field) { VisitField(field); }
   virtual void VisitEnumField(const ROOT::REnumField &field) { VisitField(field); }
   virtual void VisitFloatField(const ROOT::RField<float> &field) { VisitField(field); }
   virtual void VisitByteField(const ROOT::RField<std::byte> &field) { VisitField(field); }
   virtual void VisitCharField(const ROOT::RField<char> &field) { VisitField(field); }
   // We have to accept RIntegralField here because there can be multiple basic types that map to the same fixed-width
   // integer type; for example on 64-bit Unix systems, both long and long long map to std::int64_t.
   virtual void VisitInt8Field(const ROOT::RIntegralField<std::int8_t> &field) { VisitField(field); }
   virtual void VisitInt16Field(const ROOT::RIntegralField<std::int16_t> &field) { VisitField(field); }
   virtual void VisitInt32Field(const ROOT::RIntegralField<std::int32_t> &field) { VisitField(field); }
   virtual void VisitInt64Field(const ROOT::RIntegralField<std::int64_t> &field) { VisitField(field); }
   virtual void VisitNullableField(const ROOT::RNullableField &field) { VisitField(field); }
   virtual void VisitStringField(const ROOT::RField<std::string> &field) { VisitField(field); }
   virtual void VisitUInt8Field(const ROOT::RIntegralField<std::uint8_t> &field) { VisitField(field); }
   virtual void VisitUInt16Field(const ROOT::RIntegralField<std::uint16_t> &field) { VisitField(field); }
   virtual void VisitUInt32Field(const ROOT::RIntegralField<std::uint32_t> &field) { VisitField(field); }
   virtual void VisitUInt64Field(const ROOT::RIntegralField<std::uint64_t> &field) { VisitField(field); }
   virtual void VisitVectorField(const ROOT::RVectorField &field) { VisitField(field); }
   virtual void VisitVectorBoolField(const ROOT::RField<std::vector<bool>> &field) { VisitField(field); }
   virtual void VisitRVecField(const ROOT::RRVecField &field) { VisitField(field); }
}; // class RFieldVisitor

} // namespace Detail

namespace Internal {

// clang-format off
/**
\class ROOT::Internal::RPrepareVisitor
\ingroup NTuple
\brief Visitor used for a pre-processing run to collect information needed by another visitor class.

 Currently used for RPrintSchemaVisitor in RNTupleReader::PrintInfo() to collect information about levels, max depth etc.
*/
// clang-format on
class RPrepareVisitor : public Detail::RFieldVisitor {
private:
   unsigned int fDeepestLevel = 1;
   unsigned int fNumFields = 1;

public:
   RPrepareVisitor() = default;
   void VisitField(const ROOT::RFieldBase &field) final;
   void VisitFieldZero(const ROOT::RFieldZero &field) final;

   unsigned int GetDeepestLevel() const { return fDeepestLevel; }
   unsigned int GetNumFields() const { return fNumFields; }
};

// clang-format off
/**
\class ROOT::Internal::RPrintSchemaVisitor
\ingroup NTuple
\brief Contains settings for printing and prints a summary of an RField instance.

This visitor is used by RNTupleReader::PrintInfo()
*/
// clang-format on
class RPrintSchemaVisitor : public Detail::RFieldVisitor {
private:
   /// Where to write the printout to
   std::ostream &fOutput;
   /// To render the output, use an asterix (*) by default to draw table lines and boundaries
   char fFrameSymbol;
   /// Indicates maximal number of allowed characters per line
   int fWidth;
   int fDeepestLevel;
   int fNumFields;
   int fAvailableSpaceKeyString;
   int fAvailableSpaceValueString;
   int fFieldNo = 1;
   std::string fTreePrefix;
   std::string fFieldNoPrefix;

public:
   RPrintSchemaVisitor(std::ostream &out = std::cout, char frameSymbol = '*', int width = 80, int deepestLevel = 1,
                       int numFields = 1)
      : fOutput{out}, fFrameSymbol{frameSymbol}, fWidth{width}, fDeepestLevel{deepestLevel}, fNumFields{numFields}
   {
      SetAvailableSpaceForStrings();
   }
   /// Prints summary of Field
   void VisitField(const ROOT::RFieldBase &field) final;
   void VisitFieldZero(const ROOT::RFieldZero &fieldZero) final;
   void SetFrameSymbol(char s) { fFrameSymbol = s; }
   void SetWidth(int w) { fWidth = w; }
   void SetDeepestLevel(int d);
   void SetNumFields(int n);
   /// Computes how many characters should be placed between the frame symbol and ':' for left and right side of ':' for
   /// visually pleasing output.
   // E.g.
   // * Field 1       : vpx (std::vector<float>)                                     *
   // * |__Field 1.1  : vpx/vpx (float)                                              *
   // int fAvailableSpaceKeyString (num characters on left side between "* " and " : ")
   //    deepestlevel here is 2 (1.1 is deepest and has 2 numbers).
   //    For every additional level an additional "|_" and ".1" (4 characters) is added, so the number of required
   //    additional characters is 4 * fDeepestLevel. For level 1, 8 characters are required ("Field 1 "), so an
   //    additional + 4 is added. To account for cases where the total number of fields is not a single digit number and
   //    more space is required to output big numbers, fNumFields is incorporated into the calculation. To make sure
   //    that there is still enough space for the right side of " : ", an std::min comparision with fWidth - 15 is done.
   // int fAvailableSpaceValueString(num characters on right side between " : " and '*')
   //    The 6 subtracted characters are "* " (2) in the beginning,  " : " (3) and '*' (1) on the far right.
   void SetAvailableSpaceForStrings()
   {
      fAvailableSpaceKeyString =
         std::min(4 * fDeepestLevel + 4 + static_cast<int>(std::to_string(fNumFields).size()), fWidth - 15);
      fAvailableSpaceValueString = fWidth - 6 - fAvailableSpaceKeyString;
   }
};

// clang-format off
/**
\class ROOT::Internal::RPrintValueVisitor
\ingroup NTuple
\brief Renders a JSON value corresponding to the field.
*/
// clang-format on
class RPrintValueVisitor : public Detail::RFieldVisitor {
public:
   struct RPrintOptions {
      bool fPrintSingleLine;
      bool fPrintName;

      RPrintOptions() : fPrintSingleLine(false), fPrintName(true) {}
   };

private:
   ROOT::RFieldBase::RValue fValue;
   /// The output is directed to fOutput which may differ from std::cout.
   std::ostream &fOutput;
   unsigned int fLevel;
   RPrintOptions fPrintOptions;

   void PrintIndent();
   void PrintName(const ROOT::RFieldBase &field);
   void PrintCollection(const ROOT::RFieldBase &field);
   void PrintRecord(const ROOT::RFieldBase &field);

public:
   RPrintValueVisitor(ROOT::RFieldBase::RValue value, std::ostream &output, unsigned int level = 0,
                      RPrintOptions options = RPrintOptions())
      : fValue(value), fOutput{output}, fLevel(level), fPrintOptions(options)
   {
   }

   void VisitField(const ROOT::RFieldBase &field) final;

   void VisitBoolField(const ROOT::RField<bool> &field) final;
   void VisitDoubleField(const ROOT::RField<double> &field) final;
   void VisitFloatField(const ROOT::RField<float> &field) final;
   void VisitByteField(const ROOT::RField<std::byte> &field) final;
   void VisitCharField(const ROOT::RField<char> &field) final;
   void VisitInt8Field(const ROOT::RIntegralField<std::int8_t> &field) final;
   void VisitInt16Field(const ROOT::RIntegralField<std::int16_t> &field) final;
   void VisitInt32Field(const ROOT::RIntegralField<std::int32_t> &field) final;
   void VisitInt64Field(const ROOT::RIntegralField<std::int64_t> &field) final;
   void VisitStringField(const ROOT::RField<std::string> &field) final;
   void VisitUInt8Field(const ROOT::RIntegralField<std::uint8_t> &field) final;
   void VisitUInt16Field(const ROOT::RIntegralField<std::uint16_t> &field) final;
   void VisitUInt32Field(const ROOT::RIntegralField<std::uint32_t> &field) final;
   void VisitUInt64Field(const ROOT::RIntegralField<std::uint64_t> &field) final;

   void VisitCardinalityField(const ROOT::RCardinalityField &field) final;
   void VisitArrayField(const ROOT::RArrayField &field) final;
   void VisitArrayAsRVecField(const ROOT::RArrayAsRVecField &field) final;
   void VisitClassField(const ROOT::RClassField &field) final;
   void VisitTObjectField(const ROOT::RField<TObject> &field) final;
   void VisitStreamerField(const ROOT::RStreamerField &field) final;
   void VisitRecordField(const ROOT::RRecordField &field) final;
   void VisitProxiedCollectionField(const ROOT::RProxiedCollectionField &field) final;
   void VisitVectorField(const ROOT::RVectorField &field) final;
   void VisitVectorBoolField(const ROOT::RField<std::vector<bool>> &field) final;
   void VisitRVecField(const ROOT::RRVecField &field) final;
   void VisitBitsetField(const ROOT::RBitsetField &field) final;
   void VisitNullableField(const ROOT::RNullableField &field) final;
   void VisitEnumField(const ROOT::REnumField &field) final;
   void VisitAtomicField(const ROOT::RAtomicField &field) final;
};

// clang-format off
/**
\class ROOT::Internal::RNTupleFormatter
\ingroup NTuple
\brief Contains helper functions for RNTupleReader::PrintInfo() and RPrintSchemaVisitor::VisitField()

The functions in this class format strings which are displayed by RNTupleReader::PrintInfo() and RNTupleReader::Show().
*/
// clang-format on
class RNTupleFormatter {
public:
   // Can abbreviate long strings, e.g. ("ExampleString" , space= 8) => "Examp..."
   static std::string FitString(const std::string &str, int availableSpace);
};

} // namespace Internal
} // namespace ROOT

#endif
