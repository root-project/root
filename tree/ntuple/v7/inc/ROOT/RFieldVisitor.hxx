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
\class ROOT::Experimental::Detail::RNTupleVisitor
\ingroup NTuple
\brief Abstract base class for classes implementing the visitor design pattern.
     
 RNTupleVisitor::VisitField() is invoked by RFieldBase::AcceptVisitor(). VisitField() is inherited for instance by the RPrintVisitor class. The RFieldBase Class and classes which inherit from it will be visited.
*/
// clang-format on
class RNTupleVisitor {
public:
   virtual void VisitField(const Detail::RFieldBase &field, int level) = 0;
   virtual void VisitRootField(const RFieldRoot &field, int level) = 0;
   virtual void VisitArrayField(const RFieldArray &field, int level) { VisitField(field, level); }
   virtual void VisitBoolField(const RField<bool> &field, int level) { VisitField(field, level); }
   virtual void VisitBoolVecField(const RField<std::vector<bool>> &field, int level) { VisitField(field, level); }
   virtual void VisitClassField(const RFieldClass &field, int level) { VisitField(field, level); }
   virtual void VisitClusterSizeField(const RField<ClusterSize_t> &field, int level) { VisitField(field, level); }
   virtual void VisitDoubleField(const RField<double> &field, int level) { VisitField(field, level); }
   virtual void VisitFloatField(const RField<float> &field, int level) { VisitField(field, level); }
   virtual void VisitIntField(const RField<int> &field, int level) { VisitField(field, level); }
   virtual void VisitStringField(const RField<std::string> &field, int level) { VisitField(field, level); }
   virtual void VisitUIntField(const RField<std::uint32_t> &field, int level) { VisitField(field, level); }
   virtual void VisitUInt64Field(const RField<std::uint64_t> &field, int level) { VisitField(field, level); }
   virtual void VisitUInt8Field(const RField<std::uint8_t> &field, int level) { VisitField(field, level); }
   virtual void VisitVectorField(const RFieldVector &field, int level) { VisitField(field, level); }
};
} // namespace Detail
// clang-format off
/**
\class ROOT::Experimental::RPrepareVisitor
\ingroup NTuple
\brief Visitor used for a prepare run to collect information needed by another visitor class.

 Currently used for RPrintVisitor in RNTupleReader::Print() to collect information about levels, maximal depth etc.
*/
// clang-format on
class RPrepareVisitor : public Detail::RNTupleVisitor {
private:
   int fDeepestLevel;
   int fNumFields;

public:
   RPrepareVisitor(int deepestLevel = 0, int numFields = 0) : fDeepestLevel{deepestLevel}, fNumFields{numFields} {}
   void VisitField(const Detail::RFieldBase &field, int level) final;
   void VisitRootField(const RFieldRoot & /*field*/, int /*level*/) final {}
   int GetDeepestLevel() const { return fDeepestLevel; }
   int GetNumFields() const { return fNumFields; }
};

// clang-format off
/**
\class ROOT::Experimental::RPrintVisitor
\ingroup NTuple
\brief Contains settings for printing and prints a summary of an RField instance.
     
 Instances of this class are currently only invoked by RNTupleReader::Print() -> RFieldBase::AcceptVisitor()
*/
// clang-format on
class RPrintVisitor : public Detail::RNTupleVisitor {
private:
   /// Where to write the printout to
   std::ostream &fOutput;
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
   std::string MakeKeyString(const Detail::RFieldBase &field, int level);
   /// ValueString refers to the right side containing the type and name
   std::string MakeValueString(const Detail::RFieldBase &field);

public:
   RPrintVisitor(std::ostream &out = std::cout, char fillSymbol = '*', int width = 80, int deepestLevel = 1,
                 int numFields = 1)
      : fOutput{out}, fFrameSymbol{fillSymbol}, fWidth{width}, fDeepestLevel{deepestLevel}, fNumFields{numFields}
   {
      SetAvailableSpaceForStrings();
   }
   /// Prints summary of Field
   void VisitField(const Detail::RFieldBase &field, int level) final;
   void VisitRootField(const RFieldRoot & /*field*/, int /*level*/) final{};
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
\brief Traverses through an ntuple to display its entries.

Each visit outputs the entry of a single field as in a .json file. Used when RNTupleReader::Show() is called to show a single entry.
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
 * For Objects, it doesn't print any values and lets its subfields print the values.
 * For Vectors and Arrays the values are obtained from the most bottom level field where a basic data type is stored.
 *    The index from which the data should be read from the most bottom level field and how long a vector/array is is
 * obtained from the upper level vector and array fields.
 */

class RValueVisitor : public Detail::RNTupleVisitor {
private:
   RNTupleReader *fReader;
   /// The output is directed to fOutput which may differ from std::cout.
   std::ostream &fOutput;
   /// The fIndex-th element should be displayed.
   std::int32_t fIndex;
   /// Used when priting the value of a std::array and std::vector field. It tells the visitor to only print the value
   /// of the subfield and not create a new line for each element in the array/vector.
   bool fPrintOnlyValue;
   /// When printing the contents of a std::array or std::vector, this index is used to get the values in its itemField.
   std::size_t fCollectionIndex;

public:
   RValueVisitor(std::ostream &output, RNTupleReader *reader, std::int32_t index, bool onlyValue,
                 std::size_t collectionIndex)
      : fReader{reader}, fOutput{output}, fIndex{index}, fPrintOnlyValue{onlyValue}, fCollectionIndex{collectionIndex}
   {
   }
   void VisitField(const Detail::RFieldBase &field, int level) final;
   void VisitRootField(const RFieldRoot & /*fField*/, int /*level*/) final {}
   void VisitArrayField(const RFieldArray &field, int level) final;
   void VisitBoolField(const RField<bool> &field, int level) final;
   void VisitBoolVecField(const RField<std::vector<bool>> &field, int level) final;
   void VisitClassField(const RFieldClass &field, int level) final;
   void VisitClusterSizeField(const RField<ClusterSize_t> &field, int level) final;
   void VisitDoubleField(const RField<double> &field, int level) final;
   void VisitFloatField(const RField<float> &field, int level) final;
   void VisitIntField(const RField<int> &field, int level) final;
   void VisitStringField(const RField<std::string> &field, int level) final;
   void VisitUIntField(const RField<std::uint32_t> &field, int level) final;
   void VisitUInt64Field(const RField<std::uint64_t> &field, int level) final;
   void VisitUInt8Field(const RField<std::uint8_t> &field, int level) final;
   void VisitVectorField(const RFieldVector &field, int level) final;
   std::ostream &GetOutput() { return fOutput; }
   /// Get startIndex from next non-vector/non-array itemfield
   void SetCollectionIndex(const Detail::RFieldBase &field);
   // Necessary to convert RClusterIndexes obtained from RFieldVector::GetCollectionInfo()
   std::size_t ConvertClusterIndexToGlobalIndex(RClusterIndex cluterIndex) const;
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleFormatter
\ingroup NTuple
\brief Contains helper functions for RNTupleReader::PrintInfo() and RPrintVisitor::VisitField()
    
 The functions in this class format strings which are displayed when RNTupleReader::PrintInfo() or RNTupleReader::Show() is called.
*/
// clang-format on
class RNTupleFormatter {
public:
   static std::string FieldHierarchy(const Detail::RFieldBase &field);
   static std::string FitString(const std::string &str, int availableSpace);
   static std::string HierarchialFieldOrder(const Detail::RFieldBase &field);
};

} // namespace Experimental
} // namespace ROOT

#endif
