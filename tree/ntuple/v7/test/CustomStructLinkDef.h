#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class CustomStruct+;
#pragma link C++ class DerivedA+;
#pragma link C++ class DerivedA2+;
#pragma link C++ class DerivedB+;
#pragma link C++ class DerivedC+;
#pragma link C++ class StructWithArrays + ;
#pragma link C++ class TestEBO+;

#pragma link C++ class IAuxSetOption+;
#pragma link C++ class PackedParameters+;
#pragma link C++ class PackedContainer<int>+;

#pragma link C++ class ComplexStruct+;

#pragma link C++ class BaseOfStructWithEnums + ;
#pragma link C++ class StructWithEnums + ;

#pragma link C++ class StructUsingCollectionProxy<char> + ;
#pragma link C++ class StructUsingCollectionProxy<float> + ;
#pragma link C++ class StructUsingCollectionProxy<CustomStruct> + ;
#pragma link C++ class StructUsingCollectionProxy<StructUsingCollectionProxy<float>> + ;
#pragma link C++ class StructUsingCollectionProxy<int> + ;

#pragma link C++ class TrivialTraitsBase + ;
#pragma link C++ class TrivialTraits + ;
#pragma link C++ class TransientTraits + ;
#pragma link C++ class VariantTraitsBase + ;
#pragma link C++ class VariantTraits + ;
#pragma link C++ class StringTraits + ;
#pragma link C++ class ConstructorTraits + ;
#pragma link C++ class DestructorTraits + ;

#endif
