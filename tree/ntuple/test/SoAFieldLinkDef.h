#ifdef __CLING__

#pragma link C++ class Record+;
#pragma link C++ options=rntupleSoARecord(Record) class SoA+;

#pragma link C++ options=rntupleSoARecord(xyz) class SoAUnknownRecord+;
#pragma link C++ options=rntupleSoARecord(Record) class SoAVersionMismatch+;

#pragma link C++ class RecordSimple+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimple+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleSwapped+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleBadArray+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleBadType+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleUnexpectedMember+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleMissingMember+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleWrongMember+;

#pragma link C++ class ComplexMember+;
#pragma link C++ class RecordComplex+;
#pragma link C++ options=rntupleSoARecord(RecordComplex) class SoAComplex+;

#pragma link C++ class RecordProperties+;
#pragma link C++ class RecordDot+;
#pragma link C++ options=rntupleSoARecord(RecordProperties) class SoAProperties+;
#pragma link C++ options=rntupleSoARecord(RecordDot) class SoADot+;
#pragma link C++ options=rntupleSoARecord(RecordDot) class SoADotBadNestedType+;

#pragma link C++ class RecordBase+;
#pragma link C++ class RecordDerived+;
#pragma link C++ class RecordDerivedMulti+;
#pragma link C++ options=rntupleSoARecord(RecordBase) class SoABase+;
#pragma link C++ options=rntupleSoARecord(RecordDerived) class SoADerived+;
#pragma link C++ options=rntupleSoARecord(RecordDerivedMulti) class SoADerivedMulti+;
#pragma link C++ options=rntupleSoARecord(RecordDerived) class SoADerivedFail1+;
#pragma link C++ options=rntupleSoARecord(RecordDerived) class SoADerivedFail2+;

#pragma link C++ class RecordBaseOld+;
#pragma link C++ class RecordIntermediateOld+;
#pragma link C++ class RecordLeafOld+;
#pragma link C++ options=rntupleSoARecord(RecordBaseOld) class SoABaseOld+;
#pragma link C++ options=rntupleSoARecord(RecordIntermediateOld) class SoAIntermediateOld+;
#pragma link C++ options=rntupleSoARecord(RecordLeafOld) class SoALeafOld+;
#pragma link C++ class RecordBaseNew+;
#pragma link C++ class RecordIntermediateNew+;
#pragma link C++ class RecordLeafNew+;
#pragma link C++ options=rntupleSoARecord(RecordBaseNew) class SoABaseNew+;
#pragma link C++ options=rntupleSoARecord(RecordIntermediateNew) class SoAIntermediateNew+;
#pragma link C++ options=rntupleSoARecord(RecordLeafNew) class SoALeafNew+;

#pragma read sourceClass = "RecordBaseOld" targetClass = "RecordBaseNew" version = "[1-]"
#pragma read sourceClass = "RecordIntermediateOld" targetClass = "RecordIntermediateNew" version = "[1-]"
#pragma read sourceClass = "RecordLeafOld" targetClass = "RecordLeafNew" version = "[1-]"
#pragma read sourceClass = "SoABaseOld" targetClass = "SoABaseNew" version = "[1-]"
#pragma read sourceClass = "SoAIntermediateOld" targetClass = "SoAIntermediateNew" version = "[1-]"
#pragma read sourceClass = "SoALeafOld" targetClass = "SoALeafNew" version = "[1-]"

#pragma link C++ class RecordNested+;
#pragma link C++ class RecordOuter+;
#pragma link C++ options=rntupleSoARecord(RecordNested) class SoANested+;
#pragma link C++ options=rntupleSoARecord(RecordOuter) class SoAOuter+;

#pragma read sourceClass="SoANested" version="[1-]" targetClass="SoANested" source="" target="" \
   code="{ newObj->fInner *= 2.; }"
#pragma read sourceClass="SoAOuter" version="[1-]" targetClass="SoAOuter" source="" target="" \
   code="{ newObj->fOuter *= 4.; }"

#endif // __CLING__
