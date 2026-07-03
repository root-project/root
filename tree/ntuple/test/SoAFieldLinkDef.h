#ifdef __CLING__

#pragma link C++ class Record+;
#pragma link C++ options=rntupleSoARecord(Record) class SoA+;

#pragma link C++ options=rntupleSoARecord(xyz) class SoAUnknownRecord+;
#pragma link C++ options=rntupleSoARecord(Record) class SoAVersionMismatch+;

#pragma link C++ class RecordBase+;
#pragma link C++ class RecordDerived+;
#pragma link C++ options=rntupleSoARecord(RecordBase) class SoABase+;
#pragma link C++ options=rntupleSoARecord(RecordDerived) class SoAOnDerivedRecord+;
#pragma link C++ options=rntupleSoARecord(RecordBase) class SoADerivedOnBaseRecord+;

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

#endif // __CLING__
