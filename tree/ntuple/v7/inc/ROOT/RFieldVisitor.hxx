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
}

// clang-format off
/**
\class ROOT::Experimental::RNTupleVisitor
\ingroup NTuple
\brief Abstract base class for classes implementing the visitor design pattern.
     
 RNTupleVisitor::visitField() is invoked by RFieldBase::AcceptVisitor(). visitField() is inherited for instance by the RPrintVisitor class.
*/
// clang-format on
class RNTupleVisitor {
public:
   virtual void visitField(const Detail::RFieldBase &fField, int level) = 0;
   virtual void visitRootField(const RFieldRoot &fField, int level) = 0;
};
   
// clang-format off
/**
\class ROOT::Experimental::RPrepareVisitor
\ingroup NTuple
\brief Visitor used for a prepare run to collect information needed by another visitorclass.

 Currently used for RPrintVisitor in RNTupleReader::Print() to collect information about levels, maximal depth etc.
*/
// clang-format on
   
class RPrepareVisitor: public RNTupleVisitor {
private:
   int fDeepestLevel;
   int fNumFields;
public:
   RPrepareVisitor(int deepestLevel=0, int numFields=0): fDeepestLevel{deepestLevel}, fNumFields{numFields} { }
   void visitField(const Detail::RFieldBase &/*field*/, int level) override;
   void visitRootField(const RFieldRoot &/*field*/, int /*level*/) override { }
   int getDeepestLevel() const {return fDeepestLevel;}
   int getNumFields() const {return fNumFields;}
};

// clang-format off
/**
\class ROOT::Experimental::RPrintVisitor
\ingroup NTuple
\brief Contains settings for printing and prints a summary of an RField instance.
     
 Instances of this class are currently only invoked by RNTupleReader::Print() -> RFieldBase::AcceptVisitor()
*/
// clang-format on

class RPrintVisitor : public RNTupleVisitor {
private:
   /// Holds std::ostream which is used for holding the printed output
   std::ostream &fOutput;
   char fFrameSymbol;
   /// Indicates maximal number of allowed characters per line
   int fWidth;
   int fDeepestLevel;
   int fNumFields;
   int fAvailableSpaceKeyString;
   int fAvailableSpaceValueString;
   /// Keeps track when | is used for the tree-like structure.
   std::vector<bool> fFlagforVerticalLines;
   /// KeyString refers to the left side containing the word "Field" and its hierarchial order
   std::string KeyString(const Detail::RFieldBase &field, int level);
   /// ValueString refers to the right side containing the type and name
   std::string ValueString(const Detail::RFieldBase &field);
public:
   RPrintVisitor(std::ostream &out = std::cout, char fillSymbol = '*', int width = 80, int deepestLevel = 1, int numFields = 1)
   : fOutput{out}, fFrameSymbol{fillSymbol}, fWidth{width}, fDeepestLevel{deepestLevel}, fNumFields{numFields}
   { }
   /// Prints summary of Field
   void visitField(const Detail::RFieldBase &field, int level) override;
   void visitRootField(const RFieldRoot &/*field*/, int /*level*/) override { };
   void SetFrameSymbol(char s) {fFrameSymbol = s;}
   void SetWidth(int w) {fWidth = w;}
   void SetDeepestLevel(int d) {fDeepestLevel = d;}
   void SetNumFields(int n) {fNumFields = n;}
   void SetAvailableSpaceForStrings() {
      fAvailableSpaceKeyString = std::min(4*fDeepestLevel+4+static_cast<int>(std::to_string(fNumFields).size()), fWidth-15);
      fAvailableSpaceValueString = fWidth - 6 - fAvailableSpaceKeyString;
   }
   void ResizeFlagVec() {fFlagforVerticalLines.resize(fDeepestLevel-1);}
   void ResizeFlagVec(int size) {fFlagforVerticalLines.resize(size);}
   
};
   


std::string CutStringAndAddEllipsisIfNeeded(const std::string &toCut, int maxAvailableSpace);
} // namespace Experimental
} // namespace ROOT

#endif
