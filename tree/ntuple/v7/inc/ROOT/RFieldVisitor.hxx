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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {
class RFieldRoot;
namespace Detail {
class RFieldBase;


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
class RPrepareVisitor: public Detail::RNTupleVisitor {
private:
   int fDeepestLevel;
   int fNumFields;
public:
   RPrepareVisitor(int deepestLevel=0, int numFields=0): fDeepestLevel{deepestLevel}, fNumFields{numFields} { }
   void VisitField(const Detail::RFieldBase &field, int level) final;
   void VisitRootField(const RFieldRoot &/*field*/, int /*level*/) final { }
   int GetDeepestLevel() const {return fDeepestLevel;}
   int GetNumFields() const {return fNumFields;}
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
   // *     |__Field
   std::vector<bool> fFlagForVerticalLines;
   /// KeyString refers to the left side containing the word "Field" and its hierarchial order
   std::string MakeKeyString(const Detail::RFieldBase &field, int level);
   /// ValueString refers to the right side containing the type and name
   std::string MakeValueString(const Detail::RFieldBase &field);
public:
   RPrintVisitor(std::ostream &out = std::cout, char fillSymbol = '*', int width = 80, int deepestLevel = 1, int numFields = 1)
   : fOutput{out}, fFrameSymbol{fillSymbol}, fWidth{width}, fDeepestLevel{deepestLevel}, fNumFields{numFields}
   {SetAvailableSpaceForStrings();}
   /// Prints summary of Field
   void VisitField(const Detail::RFieldBase &field, int level) final;
   void VisitRootField(const RFieldRoot &/*field*/, int /*level*/) final { };
   void SetFrameSymbol(char s) {fFrameSymbol = s;}
   void SetWidth(int w) {fWidth = w;}
   void SetDeepestLevel(int d);
   void SetNumFields(int n);
   /// Computes how many characters should be placed between the frame symbol and ':' for left and right side of ':' for visually pleasing output.
   // E.g.
   // * Field 1       : vpx (std::vector<float>)                                     *
   // * |__Field 1.1  : vpx/vpx (float)                                              *
   // int fAvailableSpaceKeyString (num characters on left side between "* " and " : ")
   //    deepestlevel here is 2 (1.1 is deepest and has 2 numbers).
   //    For every additional level an additional "|_" and ".1" (4 characters) is added, so the number of required additional characters is 4 * fDeepestLevel.
   //    For level 1, 8 characters are required ("Field 1 "), so an additional + 4 is added.
   //    To account for cases where the total number of fields is not a single digit number and more space is required to output big numbers, fNumFields is incorporated into the calculation.
   //    To make sure that there is still enough space for the right side of " : ", an std::min comparision with fWidth - 15 is done.
   // int fAvailableSpaceValueString(num characters on right side between " : " and '*')
   //    The 6 subtracted characters are "* " (2) in the beginning,  " : " (3) and '*' (1) on the far right.
   void SetAvailableSpaceForStrings() {
      fAvailableSpaceKeyString = std::min(4 * fDeepestLevel + 4 + static_cast<int>(std::to_string(fNumFields).size()), fWidth - 15);
      fAvailableSpaceValueString = fWidth - 6 - fAvailableSpaceKeyString;
   }
};

   
// clang-format off
/**
\class ROOT::Experimental::RNTupleFormatter
\ingroup NTuple
\brief Contains helper functions for RNTupleReader::PrintInfo() and RPrintVisitor::VisitField()
    
 The functions in this class format strings which are displayed when RNTupleReader::PrinfInfo() is called.
*/
// clang-format on
class RNTupleFormatter {
public:
   static std::string FitString(const std::string &str, int availableSpace);
   static std::string HierarchialFieldOrder(const Detail::RFieldBase &field);
};

} // namespace Experimental
} // namespace ROOT

#endif
