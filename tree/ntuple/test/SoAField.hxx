#ifndef ROOT_RNTuple_Test_SoAField
#define ROOT_RNTuple_Test_SoAField

#include <ROOT/RVec.hxx>

#include <Rtypes.h>

#include <vector>

struct Record {
   ClassDefNV(Record, 2);
};

struct SoA {
   ClassDefNV(SoA, 2);
};

struct SoAUnknownRecord {
   ClassDefNV(SoAUnknownRecord, 2);
};

struct SoAVersionMismatch {
   ClassDefNV(SoAVersionMismatch, 4);
};

struct RecordSimple {
   float fX;
   float fY;
   ClassDefNV(RecordSimple, 2);
};

struct SoASimple {
   ROOT::RVec<float> fX;
   ROOT::RVec<float> fY;
   ClassDefNV(SoASimple, 2);
};

struct SoASimpleSwapped {
   ROOT::RVec<float> fY;
   ROOT::RVec<float> fX;
   ClassDefNV(SoASimpleSwapped, 2);
};

struct SoASimpleBadArray {
   ROOT::RVec<float> fX;
   float fY[3];
   ClassDefNV(SoASimpleBadArray, 2);
};

struct SoASimpleBadType {
   std::vector<float> fX;
   std::vector<float> fY;
   ClassDefNV(SoASimpleBadType, 2);
};

struct SoASimpleUnexpectedMember {
   ROOT::RVec<float> fX;
   ROOT::RVec<float> fY;
   ROOT::RVec<float> fZ;
   ClassDefNV(SoASimpleUnexpectedMember, 2);
};

struct SoASimpleMissingMember {
   ROOT::RVec<float> fX;
   ClassDefNV(SoASimpleMissingMember, 2);
};

struct SoASimpleWrongMember {
   ROOT::RVec<float> fX;
   ROOT::RVec<Double32_t> fY;
   ClassDefNV(SoASimpleWrongMember, 2);
};

/// class with non-trivial constructor and destructor
struct ComplexMember {
   inline static int gNCallConstructor = 0;
   inline static int gNCallDestructor = 0;

   ComplexMember() { gNCallConstructor++; }
   ~ComplexMember() { gNCallDestructor++; }

   ClassDefNV(ComplexMember, 2);
};

struct RecordComplex {
   ComplexMember fA;

   ClassDefNV(RecordComplex, 2);
};

struct SoAComplex {
   ROOT::RVec<ComplexMember> fA;

   ClassDefNV(SoAComplex, 2);
};

struct RecordProperties {
   int fColor;
   float fSize;

   ClassDefNV(RecordProperties, 2);
};

struct SoAProperties {
   ROOT::RVec<int> fColor;
   ROOT::RVec<float> fSize;

   ClassDefNV(SoAProperties, 2);
};

struct RecordDot {
   float fX;
   float fY;
   RecordProperties fProperties;

   ClassDefNV(RecordDot, 2);
};

struct SoADot {
   ROOT::RVec<float> fX;
   ROOT::RVec<float> fY;
   SoAProperties fProperties;

   ClassDefNV(SoADot, 2);
};

struct SoADotBadNestedType {
   ROOT::RVec<float> fX;
   ROOT::RVec<float> fY;
   SoA fProperties;

   ClassDefNV(SoADotBadNestedType, 2);
};

struct RecordBase {
   float fBase;
   ClassDefNV(RecordBase, 2);
};

struct SoABase {
   ROOT::RVec<float> fBase;
   ClassDefNV(SoABase, 2);
};

struct RecordDerived : public RecordBase {
   float fDerived;
   ClassDefNV(RecordDerived, 2);
};

struct SoADerived : public SoABase {
   ROOT::RVec<float> fDerived;
   ClassDefNV(SoADerived, 2);
};

struct RecordDerivedMulti : public RecordDerived, RecordDot {
   float fMulti;
   ClassDefNV(RecordDerivedMulti, 2);
};

struct SoADerivedMulti : public SoADerived, SoADot {
   ROOT::RVec<float> fMulti;
   ClassDefNV(SoADerivedMulti, 2);
};

struct SoADerivedFail1 : public RecordDerived {
   ClassDefNV(SoADerivedFail1, 2);
};

struct SoADerivedFail2 : public SoABase {
   ClassDefNV(SoADerivedFail2, 2);
};

struct RecordBaseOld {
   float fBase;
   ClassDefNV(RecordBaseOld, 2);
};

struct SoABaseOld {
   ROOT::RVec<float> fBase;
   ClassDefNV(SoABaseOld, 2);
};

struct RecordIntermediateOld : public RecordBaseOld {
   float fIntermediate;
   ClassDefNV(RecordIntermediateOld, 2);
};

struct SoAIntermediateOld : public SoABaseOld {
   ROOT::RVec<float> fIntermediate;
   ClassDefNV(SoAIntermediateOld, 2);
};

struct RecordLeafOld : public RecordIntermediateOld {
   float fLeaf;
   ClassDefNV(RecordLeafOld, 2);
};

struct SoALeafOld : public SoAIntermediateOld {
   ROOT::RVec<float> fLeaf;
   ClassDefNV(SoALeafOld, 2);
};

struct RecordBaseNew {
   float fBase;
   float fNew;
   ClassDefNV(RecordBaseNew, 2);
};

struct SoABaseNew {
   ROOT::RVec<float> fBase;
   ROOT::RVec<float> fNew;
   ClassDefNV(SoABaseNew, 2);
};

struct RecordIntermediateNew : public RecordBaseNew {
   float fIntermediate;
   ClassDefNV(RecordIntermediateNew, 2);
};

struct SoAIntermediateNew : public SoABaseNew {
   ROOT::RVec<float> fIntermediate;
   ClassDefNV(SoAIntermediateNew, 2);
};

struct RecordLeafNew : public RecordIntermediateNew {
   float fLeaf;
   ClassDefNV(RecordLeafNew, 2);
};

struct SoALeafNew : public SoAIntermediateNew {
   ROOT::RVec<float> fLeaf;
   ClassDefNV(SoALeafNew, 2);
};

struct RecordNested {
   float fInner;
   ClassDefNV(RecordNested, 2);
};

struct SoANested {
   ROOT::RVec<float> fInner;
   ClassDefNV(SoANested, 2);
};

struct RecordOuter {
   float fOuter;
   RecordNested fNested;
   ClassDefNV(RecordOuter, 2);
};

struct SoAOuter {
   ROOT::RVec<float> fOuter;
   SoANested fNested;
   ClassDefNV(SoAOuter, 2);
};

#endif // ROOT_RNTuple_Test_SoAField
