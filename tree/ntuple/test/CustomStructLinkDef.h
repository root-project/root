#ifdef __CLING__

#pragma link C++ enum CustomEnum;
#pragma link C++ enum CustomEnumInt8;
#pragma link C++ enum CustomEnumUInt8;
#pragma link C++ enum CustomEnumInt16;
#pragma link C++ enum CustomEnumUInt16;
#pragma link C++ enum CustomEnumInt32;
#pragma link C++ enum CustomEnumUInt32;
#pragma link C++ enum CustomEnumInt64;
#pragma link C++ enum CustomEnumUInt64;

#pragma link C++ class CustomStruct+;
#pragma link C++ class DerivedA+;
#pragma link C++ class DerivedA2+;
#pragma link C++ class DerivedWithTypedef + ;
#pragma link C++ class DerivedB+;
#pragma link C++ class DerivedC+;
#pragma link C++ class StructWithArrays + ;
#pragma link C++ class EmptyStruct + ;
#pragma link C++ class TestEBO+;
#pragma link C++ class IOConstructor+;
#pragma link C++ class LowPrecisionFloats+;

#pragma link C++ class EdmWrapper<CustomStruct> +;
#pragma link C++ class EdmHash < 1> + ;

#pragma link C++ class DataVector < int, double> + ;
#pragma link C++ class DataVector < int, float> + ;
#pragma link C++ class DataVector < bool, std::vector < unsigned int>> + ;
#pragma link C++ class DataVector < StructUsingCollectionProxy < int>, double> + ;
#pragma link C++ class DataVector < int, double> ::Inner + ;
#pragma link C++ class DataVector < int, double> ::Inner + ;
#pragma link C++ class DataVector < int, float> ::Inner + ;
#pragma link C++ class DataVector < int, float> ::Inner + ;
#pragma link C++ class DataVector < int, double> ::Nested < int, double> + ;
#pragma link C++ class DataVector < int, double> ::Nested < int, float> + ;
#pragma link C++ class DataVector < int, float> ::Nested < int, double> + ;
#pragma link C++ class DataVector < int, float> ::Nested < int, float> + ;
#pragma link C++ class InnerCV < const int, const volatile int, volatile const int, volatile int> + ;
#pragma link C++ class InnerCV < const std::vector<std::string[2]>, int, int, int> + ;
#pragma link C++ class IntegerTemplates < 0, 0> + ;
#pragma link C++ class IntegerTemplates < -1, 1> + ;
#pragma link C++ class IntegerTemplates < -2147483650ll, 9223372036854775810ull> + ;

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

#pragma link C++ options = version(3) class StructWithIORulesBase + ;
#pragma link C++ options = version(3) class StructWithTransientString + ;
#pragma link C++ options = version(3) class StructWithIORules + ;

#pragma read sourceClass = "StructWithIORulesBase" source = "float a" version = "[1-99]" targetClass = \
   "StructWithIORulesBase" target = "b" code = "{ b = onfile.a + 1.0f; }"

// This rule is ignored due to type version mismatch
#pragma read sourceClass = "StructWithIORulesBase" source = "float a" version = "[100-]" targetClass = \
   "StructWithIORulesBase" target = "b" code = "{ b = 0.0f; }"

#pragma read sourceClass = "StructWithTransientString" source = "char chars[4]" version = "[1-]" targetClass = \
   "StructWithTransientString" target = "str" include = "string" code = "{ str = std::string{onfile.chars, 4}; }"

// Whole object rule (without target member) listed first but should be executed last
// clang-format off
#pragma read sourceClass="StructWithIORules" version="[1-]" targetClass="StructWithIORules" source="" target="" \
   code="{ newObj->cDerived = 2 * newObj->c; }"
// clang-format on

#pragma read sourceClass = "StructWithIORules" source = "float a" version = "[1-]" targetClass = \
   "StructWithIORules" target = "c" code = "{ c = onfile.a + newObj->b; }"

// Conflicting type for source member
#pragma read sourceClass = "StructWithIORules" source = "double a" version = "[1-]" targetClass = \
   "StructWithIORules" target = "" code = "{ }"

// This rule uses a checksum to identify the source class
#pragma read sourceClass = "StructWithIORules" source = "" checksum = "[3494027874]" targetClass = \
   "StructWithIORules" target = "checksumA" code = "{ checksumA = 42.0; }"
// This rule will be ignored due to a checksum mismatch
#pragma read sourceClass = "StructWithIORules" source = "" checksum = "[1]" targetClass = "StructWithIORules" target = \
   "checksumB" code = "{ checksumB = 0.0; }"

#pragma link C++ options = version(3) class OldCoordinates + ;

#pragma link C++ options = version(3) class CoordinatesWithIORules + ;

#pragma read sourceClass = "CoordinatesWithIORules" source = "float fX; float fY" version = "[3]" targetClass = \
   "CoordinatesWithIORules" target = "fPhi,fR" include = "cmath" code =                                         \
      "{ fR = sqrt(onfile.fX * onfile.fX + onfile.fY * onfile.fY); fPhi = atan2(onfile.fY, onfile.fX); }"

#pragma read sourceClass = "OldCoordinates" source = "float fOldX; float fOldY" version = "[3]" targetClass = \
   "CoordinatesWithIORules" target = "fX,fY,fPhi,fR" include = "cmath" code =                                 \
      "{ fR = sqrt(onfile.fOldX * onfile.fOldX + onfile.fOldY * onfile.fOldY); \
      fPhi = atan2(onfile.fOldY, onfile.fOldX); \
      fX = onfile.fOldX; fY = onfile.fOldY; }"

#pragma link C++ options = version(3) class LowPrecisionFloatWithIORules + ;

#pragma read sourceClass = "LowPrecisionFloatWithIORules" source = "float fLast8BitsZero;" version =       \
   "[3]" targetClass = "LowPrecisionFloatWithIORules" target = "fLast8BitsZero" include = "cstring,cstdint" code = \
      "{ std::uint32_t bits; std::memcpy(&bits, &onfile.fLast8BitsZero, sizeof(bits)); \
         bits |= 137; /* placeholder for randomizing the 8 LSBs */ \
         std::memcpy(&fLast8BitsZero, &bits, sizeof(fLast8BitsZero)); }"

#pragma link C++ options = version(3) class OldName < int> + ;
#pragma link C++ options = version(3) class OldName < OldName < int>> + ;
#pragma link C++ options = version(3) class NewName < int> + ;
#pragma link C++ options = version(3) class NewName < NewName < int>> + ;
#pragma read sourceClass = "OldName<OldName<int>>" targetClass = "NewName<OldName<int>>" version = "[3]"

#pragma link C++ struct SourceStruct + ;
#pragma link C++ struct StructWithSourceStruct + ;
#pragma read sourceClass = "StructWithSourceStruct" source = "SourceStruct fSource" targetClass = \
   "StructWithSourceStruct" target = "fTransient" code =                                          \
      "{ fTransient = onfile.fSource.fValue + onfile.fSource.fTransient; }"

#pragma link C++ class Cyclic + ;
#pragma link C++ class CyclicCollectionProxy + ;
#pragma link C++ class Unsupported + ;

#pragma link C++ class BaseA + ;
#pragma link C++ class DiamondVirtualB + ;
#pragma link C++ class DiamondVirtualC + ;
#pragma link C++ class DiamondVirtualD + ;
#pragma link C++ class DuplicateBaseB + ;
#pragma link C++ class DuplicateBaseC + ;
#pragma link C++ class DuplicateBaseD + ;

#pragma link C++ class Left + ;
#pragma link C++ class DerivedFromLeftAndTObject+;

#pragma link C++ class ThrowForVariant + ;

#pragma link C++ class RelativelyLargeStruct + ;

#pragma link C++ class v1::Vector3D+;
#pragma link C++ class v1::ExampleMC+;
#pragma link C++ class v2::ExampleMC+;
#pragma read sourceClass = "v1::ExampleMC" source = "v1::Vector3D fSpin" version="[1-]" targetClass = \
   "v2::ExampleMC" target = "fHelicity" code = "{ fHelicity = onfile.fSpin.fZ; }"

#endif
