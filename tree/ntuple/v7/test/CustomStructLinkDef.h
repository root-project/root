#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class CustomStruct+;
#pragma link C++ class DerivedA+;
#pragma link C++ class DerivedA2+;
#pragma link C++ class DerivedB+;
#pragma link C++ class DerivedC+;

#pragma link C++ class IAuxSetOption+;
#pragma link C++ class PackedParameters+;
#pragma link C++ class PackedContainer<int>+;

#pragma link C++ class ComplexStruct+;

#pragma link C++ class BaseOfStructWithEnums + ;
#pragma link C++ class StructWithEnums + ;

#endif
