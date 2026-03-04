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

struct RecordBase {
   ClassDefNV(RecordBase, 2);
};

struct RecordDerived : public RecordBase {
   ClassDefNV(RecordDerived, 2);
};

struct SoABase {
   ClassDefNV(SoABase, 2);
};

struct SoAOnDerivedRecord {
   ClassDefNV(SoAOnDerivedRecord, 2);
};

struct SoADerivedOnBaseRecord : public SoABase {
   ClassDefNV(SoADerivedOnBaseRecord, 2);
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

#endif // ROOT_RNTuple_Test_SoAField
