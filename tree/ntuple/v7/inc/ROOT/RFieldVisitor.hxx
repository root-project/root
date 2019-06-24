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


//#include "ROOT/RField.hxx"

namespace ROOT {
namespace Experimental {
    //template<typename T, typename T2>
    //class RField;
    class RFieldRoot;
namespace Detail {
    class RFieldBase;
    
    }
    
    
class RNTupleVisitor {
public:
    virtual void visitField(const ROOT::Experimental::Detail::RFieldBase& fField) = 0;
    virtual void visitField(const ROOT::Experimental::RFieldRoot& fRootField) = 0;
    //virtual void visitField(ROOT::Experimental::Detail::RFieldBase* fField, int floop);
    //virtual void visitField(ROOT::Experimental::RField<std::string, void>* fField, int floop);
};
    
class RPrintVisitor: public RNTupleVisitor {
public:
    void visitField(const ROOT::Experimental::Detail::RFieldBase& fField);
    void visitField(const ROOT::Experimental::RFieldRoot& fRootField);
    //void visitField(ROOT::Experimental::Detail::RFieldBase* fField, int floop);
    //void visitField(ROOT::Experimental::RField<std::string, void>* fField, int floop);
};
    
int NumDigits(int x);
    /*
class RNTupleReader;

class VBaseNtupleVisitor {
public:
    virtual void visitNtuple(ROOT::Experimental::RNTupleReader* fReader) = 0;
};

class TNtuplePrintVisitor: public VBaseNtupleVisitor {
public:
    void visitNtuple(ROOT::Experimental::RNTupleReader* fReader);
};
    */
} // namespace Experimental
} // namespace ROOT

#endif
