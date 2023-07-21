#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ enum CustomEnum;
#pragma link C++ enum class CustomEnumInt8;
#pragma link C++ enum class CustomEnumUInt8;
#pragma link C++ enum class CustomEnumInt16;
#pragma link C++ enum class CustomEnumUInt16;
#pragma link C++ enum class CustomEnumInt32;
#pragma link C++ enum class CustomEnumUInt32;
#pragma link C++ enum class CustomEnumInt64;
#pragma link C++ enum class CustomEnumUInt64;

#pragma link C++ class CustomStruct+;
#pragma link C++ class DerivedA+;
#pragma link C++ class DerivedA2+;
#pragma link C++ class DerivedB+;
#pragma link C++ class DerivedC+;
#pragma link C++ class StructWithArrays + ;
#pragma link C++ class EmptyStruct + ;
#pragma link C++ class TestEBO+;
#pragma link C++ class IOConstructor+;
#pragma link C++ class LowPrecisionFloats+;

#pragma link C++ class EdmWrapper<CustomStruct> +;

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

#pragma link C++ class std::set<std::int64_t> +;
#pragma link C++ class std::set<std::string> +;
#pragma link C++ class std::set<float> +;
#pragma link C++ class std::set<std::set<CustomStruct>> +;
#pragma link C++ class std::set<std::set<char>> +;
#pragma link C++ class std::set<std::pair<int, CustomStruct>> +;

#pragma link C++ options = version(3) class StructWithIORulesBase + ;
#pragma link C++ options = version(3) class StructWithTransientString + ;
#pragma link C++ options = version(3) class StructWithIORules + ;

#pragma read sourceClass = "StructWithIORulesBase" source = "float a" version = "[1-99]" targetClass = \
   "StructWithIORulesBase" target = "b" code = "{ b = onfile.a + 1.0f; }"
// Including a non-transient member in `target` should issue a warning and ignore the rule; thus, `a` remains unchanged
// in the test
#pragma read sourceClass = "StructWithIORulesBase" source = "float a" version = "[1-]" targetClass = \
   "StructWithIORulesBase" target = "a" code = "{ a = 0.0f; }"
// This rule is ignored due to type version mismatch
#pragma read sourceClass = "StructWithIORulesBase" source = "float a" version = "[100-]" targetClass = \
   "StructWithIORulesBase" target = "b" code = "{ b = 0.0f; }"

#pragma read sourceClass = "StructWithTransientString" source = "char chars[4]" version = "[1-]" targetClass = \
   "StructWithTransientString" target = "str" include = "string" code = "{ str = std::string{onfile.chars, 4}; }"

#pragma read sourceClass = "StructWithIORules" source = "float a;float b" version = "[1-]" targetClass = \
   "StructWithIORules" target = "c" code = "{ c = onfile.a + onfile.b; }"

#endif
