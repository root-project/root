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

// Note: This rule has been modified to work around ROOT bug #15877.
// The original rule was `str = std::string{onfile.chars, 4};`
//
// This bug is triggered by the TClassReadRules unit test (in rfield_class.cxx) in the following way:
//   1. Upon write, RNTuple calls TClass::GetStreamerInfo() to store the streamer info of StructWithTransientString
//   2. The read rule calls TClass::GetDataMemberOffset("chars") to fill the `onfile` variable
//   3. The class doesn't find "chars" among its real data members (it's "chars[4]" in this list)
//   4. The class therefore tries to get the offset from the streamer info; the streamer info exists in
//      GetCurrentStreamerInfo() because we called TClass::GetStreamerInfo() in step 1.
//      Otherwise GetDataMemberOffset() would return 0 which happens to be correct.
//   5. Now we enter the bug:
//      - The streamer info has two elements for "chars", one with the correct offset (0),
//        one cached, with a wrong one (8)
//      - The streamer info returns the offset of the wrong data member
#pragma read sourceClass = "StructWithTransientString" source = "char chars[4]" version = "[1-]" targetClass = \
   "StructWithTransientString" target = "str" include = "string" code = "{ str = \"ROOT\"; }"

#pragma read sourceClass = "StructWithIORules" source = "float a;float b" version = "[1-]" targetClass = \
   "StructWithIORules" target = "c" code = "{ c = onfile.a + onfile.b; }"

// This rule uses a checksum to identify the source class
#pragma read sourceClass = "StructWithIORules" source = "float checksumA" checksum = "[3494027874]" targetClass = \
   "StructWithIORules" target = "checksumA" code = "{ checksumA = 42.0; }"
// This rule will be ignored due to a checksum mismatch
#pragma read sourceClass = "StructWithIORules" source = "float checksumB" checksum = "[1]" targetClass = \
   "StructWithIORules" target = "checksumB" code = "{ checksumB = 0.0; }"

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

#endif
