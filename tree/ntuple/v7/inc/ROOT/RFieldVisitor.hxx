/// \file ROOT/RFieldVisitor.hxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
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

#ifndef ROOT7_RFieldVisitor
#define ROOT7_RFieldVisitor

#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {
class RNTupleReader;
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RVisitorRank
\ingroup NTuple
\brief Describes where the visitor is located during the traversal.

Used by the visitors to act upon first/last children when iterating the tree of fields or values
*/
// clang-format on
class RVisitorRank {
private:
   /// Tells how deep the field is in the ntuple. Root field has fLevel 0, direct subfields of root have fLevel 1, etc.
   unsigned int fLevel;
   /// First child of parent has fOrder 0, the next fOrder 1, etc. If there is no parent, fOrder is -1
   int fOrder;
   /// The field itself is also included in this number.
   unsigned int fNumSiblings;
   /// Children refers to elements of fSubField
   unsigned int fNumChildren;

public:
   RVisitorRank(unsigned int lvl, unsigned int ord, unsigned int nSiblings, unsigned int nChildren)
      : fLevel(lvl), fOrder(ord), fNumSiblings(nSiblings), fNumChildren(nChildren)
   {
   }
   unsigned int GetLevel() const { return fLevel; }
   int GetOrder() const { return fOrder; }
   unsigned int GetNumSiblings() const { return fNumSiblings; }
   unsigned int GetNumChildren() const { return fNumChildren; }
   bool IsFirstSibling() const;
   bool IsLastSibling() const;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RSchemaVisitor
\ingroup NTuple
\brief Abstract base class for classes implementing the visitor design pattern.

RSchemaVisitor::VisitField() is invoked by RFieldBase::AcceptSchemaVisitor(). VisitField() is inherited for instance
by the RPrintSchemaVisitor class. The RFieldBase class and classes which inherit from it will be visited.
*/
// clang-format on
class RSchemaVisitor {
public:
   virtual void VisitField(const Detail::RFieldBase &field, const RVisitorRank &rank) = 0;
   virtual void VisitRootField(const RFieldRoot &field, const RVisitorRank &rank) = 0;
   virtual void VisitArrayField(const RFieldArray &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitBoolField(const RField<bool> &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitClassField(const RFieldClass &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitClusterSizeField(const RField<ClusterSize_t> &field, const RVisitorRank &rank) {
      VisitField(field, rank);
   }
   virtual void VisitDoubleField(const RField<double> &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitFloatField(const RField<float> &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitIntField(const RField<int> &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitStringField(const RField<std::string> &field, const RVisitorRank &rank) {
      VisitField(field, rank);
   }
   virtual void VisitUInt32Field(const RField<std::uint32_t> &field, const RVisitorRank &rank) {
      VisitField(field, rank);
   }
   virtual void VisitUInt64Field(const RField<std::uint64_t> &field, const RVisitorRank &rank) {
      VisitField(field, rank);
   }
   virtual void VisitUInt8Field(const RField<std::uint8_t> &field, const RVisitorRank &rank) {
      VisitField(field, rank);
   }
   virtual void VisitVectorField(const RFieldVector &field, const RVisitorRank &rank) { VisitField(field, rank); }
   virtual void VisitVectorBoolField(const RField<std::vector<bool>> &field, const RVisitorRank &rank) {
      VisitField(field, rank);
   }
}; // class RSchemaVisitor


// clang-format off
/**
\class ROOT::Experimental::Detail::RValueVisitor
\ingroup NTuple
\brief Abstract base class for visiting all the values of a certain field

RValueVisitor::VisitField() is invoked by RFieldBase::AcceptValueVisitor().
*/
// clang-format on
class RValueVisitor {
protected:
   RFieldValue fValue;

public:
   explicit RValueVisitor(const RFieldValue value) : fValue(value) {}
   RFieldValue GetValue() const { return fValue; }

   virtual void VisitField(const Detail::RFieldBase &field) = 0;
   virtual void VisitFloatField(const RField<float> &field) { VisitField(field); }
   virtual void VisitDoubleField(const RField<double> &field) { VisitField(field); }
   virtual void VisitVectorField(const RFieldVector &field) { VisitField(field); }
}; // class RValueVisitor

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RPrepareVisitor
\ingroup NTuple
\brief Visitor used for a pre-processing run to collect information needed by another visitor class.

 Currently used for RPrintSchemaVisitor in RNTupleReader::Print() to collect information about levels, max depth etc.
*/
// clang-format on
class RPrepareVisitor : public Detail::RSchemaVisitor {
private:
   unsigned int fDeepestLevel;
   unsigned int fNumFields;

public:
   RPrepareVisitor(unsigned int deepestLevel = 0, unsigned int numFields = 0)
      : fDeepestLevel{deepestLevel}, fNumFields{numFields} {}
   void VisitField(const Detail::RFieldBase &field, const Detail::RVisitorRank &rank) final;
   void VisitRootField(const RFieldRoot & /*field*/, const Detail::RVisitorRank & /*rank*/) final {}
   unsigned int GetDeepestLevel() const { return fDeepestLevel; }
   unsigned int GetNumFields() const { return fNumFields; }
};


// clang-format off
/**
\class ROOT::Experimental::RPrintSchemaVisitor
\ingroup NTuple
\brief Contains settings for printing and prints a summary of an RField instance.

This visitor is used by RNTupleReader::Print()
*/
// clang-format on
class RPrintSchemaVisitor : public Detail::RSchemaVisitor {
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
   /// Keeps track when | is used for the tree-like structure.
   // E.g. in
   // * | | |__Field
   // * | |__Field 2
   // * |   |__Field <- '|' in position 1, but no '|' in position 2
   // * |__Field 2.4
   // *   |__Field 2 <- no '|' in position 1
   // *     |__Field <- no '|' in position 1 and 2
   std::vector<bool> fFlagForVerticalLines;
   /// KeyString refers to the left side containing the word "Field" and its hierarchial order
   std::string MakeKeyString(const Detail::RFieldBase &field, const Detail::RVisitorRank &rank);
   /// ValueString refers to the right side containing the type and name
   std::string MakeValueString(const Detail::RFieldBase &field);

public:
   RPrintSchemaVisitor(std::ostream &out = std::cout, char frameSymbol = '*', int width = 80, int deepestLevel = 1,
                       int numFields = 1)
      : fOutput{out}, fFrameSymbol{frameSymbol}, fWidth{width}, fDeepestLevel{deepestLevel}, fNumFields{numFields}
   {
      SetAvailableSpaceForStrings();
   }
   /// Prints summary of Field
   void VisitField(const Detail::RFieldBase &field, const Detail::RVisitorRank &rank) final;
   void VisitRootField(const RFieldRoot & /*field*/, const Detail::RVisitorRank & /*rank*/) final{};
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
\class ROOT::Experimental::RValueVisitor
\ingroup NTuple
\brief Traverses through the fields of an ntuple entry the values.

Each visit outputs the entry of a single field JSON formatted. Used by RNTupleReader::Show().
*/
// clang-format on

/*
 * A few notes on why this type procedure was chosen:
 * 1. Overloading the streaming operator "<<" for array and vector would have been possible, but it wouldn't have been
 * able to overload them for objects.
 * 2. Creating a RField::GetView-function was not possible, because when objects are generated from the descriptor, the
 * RField for vectors and arrays are untemplated (RFieldArray, RFieldVector). (This would have allowed to call
 * vectorName.size() to get the length instead of doing complicated arithmetics.)
 * 3. By not using RField::Read() and RField::GenerateValue(), const_cast in not required.
 * 4. Mulitidimensional vectors could be displayed with gInterpreter but using gInterpreter for vectors would have not
 * allowed to show vectors of objects.
 *
 * How it works:
 * For every field type its appropriate visitor is called.
 * For objects, it doesn't print any values and lets its subfields print the values.
 * For vectors and arrays the values are obtained from the most bottom level field where a basic data type is stored.
 *    The index from which the data should be read from the most bottom level field and how long a vector/array is is
 * obtained from the upper level vector and array fields.
 */

class RRemoveMeVisitor : public Detail::RSchemaVisitor {
private:
   RNTupleReader *fReader;
   /// The output is directed to fOutput which may differ from std::cout.
   std::ostream &fOutput;
   /// The fIndex-th element should be displayed.
   std::int32_t fIndex;
   /// Used when printing the value of a std::array and std::vector field. It tells the visitor to only print the value
   /// of the subfield and not create a new line for each element in the array/vector.
   bool fPrintOnlyValue;
   /// When printing the contents of a std::array or std::vector, this index is used to get the values in its itemField.
   std::size_t fCollectionIndex;

public:
   RRemoveMeVisitor(std::ostream &output, RNTupleReader *reader, std::int32_t index, bool onlyValue,
                 std::size_t collectionIndex)
      : fReader{reader}, fOutput{output}, fIndex{index}, fPrintOnlyValue{onlyValue}, fCollectionIndex{collectionIndex}
   {
   }
   void VisitField(const Detail::RFieldBase &field, const Detail::RVisitorRank &rank) final;
   void VisitRootField(const RFieldRoot & /*fField*/, const Detail::RVisitorRank & /*rank*/) final {}
   void VisitArrayField(const RFieldArray &field, const Detail::RVisitorRank &rank) final;
   void VisitBoolField(const RField<bool> &field, const Detail::RVisitorRank &rank) final;
   void VisitClassField(const RFieldClass &field, const Detail::RVisitorRank &rank) final;
   void VisitClusterSizeField(const RField<ClusterSize_t> &field, const Detail::RVisitorRank &rank) final;
   void VisitDoubleField(const RField<double> &field, const Detail::RVisitorRank &rank) final;
   void VisitFloatField(const RField<float> &field, const Detail::RVisitorRank &rank) final;
   void VisitIntField(const RField<int> &field, const Detail::RVisitorRank &rank) final;
   void VisitStringField(const RField<std::string> &field, const Detail::RVisitorRank &rank) final;
   void VisitUInt32Field(const RField<std::uint32_t> &field, const Detail::RVisitorRank &rank) final;
   void VisitUInt64Field(const RField<std::uint64_t> &field, const Detail::RVisitorRank &rank) final;
   void VisitUInt8Field(const RField<std::uint8_t> &field, const Detail::RVisitorRank &rank) final;
   void VisitVectorField(const RFieldVector &field, const Detail::RVisitorRank &rank) final;
   void VisitVectorBoolField(const RField<std::vector<bool>> &field, const Detail::RVisitorRank &rank) final;
   std::ostream &GetOutput() { return fOutput; }
   /// Get startIndex from next non-vector/non-array itemfield
   void SetCollectionIndex(const Detail::RFieldBase &field);
   // Necessary to convert RClusterIndexes obtained from RFieldVector::GetCollectionInfo()
   std::size_t ConvertClusterIndexToGlobalIndex(RClusterIndex cluterIndex) const;
};

// clang-format off
/**
\class ROOT::Experimental::RPrintValueVisitor
\ingroup NTuple
\brief Renders a JSON value corresponding to the field. Used with RFieldValue::AcceptValueVisitor()
*/
// clang-format on
class RPrintValueVisitor : public Detail::RValueVisitor {
public:
   struct RPrintOptions {
      bool fPrintSingleLine;
      bool fPrintName;

      RPrintOptions() : fPrintSingleLine(false), fPrintName(true) {}
   };

private:
   /// The output is directed to fOutput which may differ from std::cout.
   std::ostream &fOutput;
   unsigned int fLevel;
   RPrintOptions fPrintOptions;

   void PrintIndent();
   void PrintName(const Detail::RFieldBase &field);

public:
   RPrintValueVisitor(Detail::RFieldValue value,
                      std::ostream &output,
                      unsigned int level = 0,
                      RPrintOptions options = RPrintOptions())
      : Detail::RValueVisitor(value), fOutput{output}, fLevel(level), fPrintOptions(options) {}

   void VisitField(const Detail::RFieldBase &field) final;
   void VisitFloatField(const RField<float> &field) final;
   void VisitVectorField(const RFieldVector &field) final;
};


// clang-format off
/**
\class ROOT::Experimental::RNTupleFormatter
\ingroup NTuple
\brief Contains helper functions for RNTupleReader::PrintInfo() and RPrintSchemaVisitor::VisitField()

The functions in this class format strings which are displayed by RNTupleReader::PrintInfo() and RNTupleReader::Show().
*/
// clang-format on
class RNTupleFormatter {
public:
   // Returns a nested field with its parent fields, like "SubFieldofRootFieldName. ... .ParentFieldName.FieldName"
   // TODO(jblomer): remove me
   static std::string FieldHierarchy(const Detail::RFieldBase &field, const Detail::RVisitorRank &rank);
   // Can abbreviate long strings, e.g. ("ExampleString" , space= 8) => "Examp..."
   static std::string FitString(const std::string &str, int availableSpace);
   // Returns std::string of form "1" or "2.1.1"
   static std::string HierarchialFieldOrder(const Detail::RFieldBase &field);
};

} // namespace Experimental
} // namespace ROOT

#endif
