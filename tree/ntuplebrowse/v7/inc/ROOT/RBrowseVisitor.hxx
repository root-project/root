/// \file ROOT/RBrowseVisitor.hxx
/// \ingroup NTupleBrowse ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-07-30
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowseVisitor
#define ROOT7_RBrowseVisitor

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>


class TBrowser;
namespace ROOT {
namespace Experimental {
class RNTupleBrowser;
   
   
// numeric data types, for which a histogram is drawn.
enum numericDatatype {
   numericDatatype_nonNumeric,   //0
   numericDatatype_float,        //1
   numericDatatype_double,       //2
   numericDatatype_Int32,        //3
   numericDatatype_UInt32,       //4
   numericDatatype_UInt64,       //5
   numericDatatype_notkLeaf,     //6
   numericDatatype_parentIsVec,  //7
   numericDatatype_noHist        //8, default one, has always this value when the first Visit-function is called.
};
// Using type void allows compatibility with unit tests. To reverse, replace void* by TBrowser*
class RBrowseVisitor: public Detail::RNTupleVisitor {
public:
   RBrowseVisitor(TBrowser* parb, RNTupleBrowser* parntplb): b{parb}, ntplb{parntplb} {}
   TBrowser* b;
   RNTupleBrowser* ntplb;
   numericDatatype fType = numericDatatype_noHist;
   void VisitField(const Detail::RFieldBase &field, int level) final;
   void VisitRootField(const RFieldRoot &/*field*/, int /*level*/) final { }
   void VisitFloatField(const RField<float> &field, int level) {
      fType = numericDatatype_float;
      VisitField(field, level);
   }
   void VisitDoubleField(const RField<double> &field, int level) {
      fType = numericDatatype_double;
      VisitField(field, level);
   }
   void VisitInt32Field(const RField<std::int32_t> &field, int level) {
      fType = numericDatatype_Int32;
      VisitField(field, level);
   }
   void VisitUInt32Field(const RField<std::uint32_t> &field, int level) {
      fType = numericDatatype_UInt32;
      VisitField(field, level);
   }
   void VisitUInt64Field(const RField<std::uint64_t> &field, int level) {
      fType = numericDatatype_UInt64;
      VisitField(field, level);
   }
};
   
} // namespace Experimental
} // namespace ROOT

#endif /* ROOT7_RBrowseVisitor */
