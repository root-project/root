#ifdef __CLING__

#pragma link C++ class Record+;
#pragma link C++ options=rntupleSoARecord(Record) class SoA+;

#pragma link C++ options=rntupleSoARecord(xyz) class SoAUnknownRecord+;

#pragma link C++ class RecordBase+;
#pragma link C++ class RecordDerived+;
#pragma link C++ options=rntupleSoARecord(RecordBase) class SoABase+;
#pragma link C++ options=rntupleSoARecord(RecordDerived) class SoAOnDerivedRecord+;
#pragma link C++ options=rntupleSoARecord(RecordBase) class SoADerivedOnBaseRecord+;

#pragma link C++ class RecordSimple+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimple+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleBadArray+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleBadType+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleUnexpectedMember+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleMissingMember+;
#pragma link C++ options=rntupleSoARecord(RecordSimple) class SoASimpleWrongMember+;

#endif // __CLING__
