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
   
   
// Indicate the type of data a field contains. It is used to deduce if a histogram for that field can/should be displayed and also how.
enum fieldDatatype {
   fieldDatatype_nonNumeric,   //0
   fieldDatatype_float,        //1
   fieldDatatype_double,       //2
   fieldDatatype_Int32,        //3
   fieldDatatype_UInt32,       //4
   fieldDatatype_UInt64,       //5
   fieldDatatype_notkLeaf,     //6
   fieldDatatype_parentIsVec,  //7
   fieldDatatype_noHist        //8, default one, has always this value when the first Visit-function is called.
};

// clang-format off
/**
\class ROOT::Experimental::RBrowseVisitor
\ingroup NTupleBrowse
\brief Visitor class which traverses fields to display them on the TBrowser.
    
RBrowseVisitor uses information about a field and creates an instance of RNTupleFieldElement or RNTupleFieldElementFolder.
*/
// clang-format on

class RBrowseVisitor: public Detail::RNTupleVisitor {
private:
   /// Is passed down to RNTupleFieldElement or RNTupleFieldElementFolder.
   TBrowser*            fBrowser;
   /// Keeps track of data type stored in field.
   fieldDatatype        fType;
   /// Used to save created instance of RNTupleFieldElement or RNTupleFieldElementFolder in RNTupleBrowser and also passed down to RNTupleFieldElement and RNTupleFieldElementFolder.
   RNTupleBrowser*      fNTupleBrowserPtr;
   
public:
   RBrowseVisitor(TBrowser* parb, RNTupleBrowser* parntplb): fBrowser{parb}, fType{fieldDatatype_noHist}, fNTupleBrowserPtr{parntplb} {}
   
   /// Creates instance of RNTupleFieldElement or RNTupleFieldElementFolder and displays it in TBrowser.
   void VisitField(const Detail::RFieldBase &field, int level) final;
   // Do nothing for RootField
   void VisitRootField(const RFieldRoot &/*field*/, int /*level*/) final { }
   void VisitFloatField(const RField<float> &field, int level) {
      fType = fieldDatatype_float;
      VisitField(field, level);
   }
   void VisitDoubleField(const RField<double> &field, int level) {
      fType = fieldDatatype_double;
      VisitField(field, level);
   }
   void VisitInt32Field(const RField<std::int32_t> &field, int level) {
      fType = fieldDatatype_Int32;
      VisitField(field, level);
   }
   void VisitUInt32Field(const RField<std::uint32_t> &field, int level) {
      fType = fieldDatatype_UInt32;
      VisitField(field, level);
   }
   void VisitUInt64Field(const RField<std::uint64_t> &field, int level) {
      fType = fieldDatatype_UInt64;
      VisitField(field, level);
   }
};
   
} // namespace Experimental
} // namespace ROOT

#endif /* ROOT7_RBrowseVisitor */
