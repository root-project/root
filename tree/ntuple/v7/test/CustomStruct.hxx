#ifndef ROOT7_RNTuple_Test_CustomStruct
#define ROOT7_RNTuple_Test_CustomStruct

#include <RtypesCore.h> // for Double32_t
#include <TRootIOCtor.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

/**
 * Used to test serialization and deserialization of classes in RNTuple with TClass
 */

enum CustomEnum { kCustomEnumVal = 7 };
// TODO(jblomer): use standard integer types for specifying the underlying width; requires TEnum fix.
enum class CustomEnumInt8 : char {};
enum class CustomEnumUInt8 : unsigned char {};
enum class CustomEnumInt16 : short int {};
enum class CustomEnumUInt16 : unsigned short int {};
enum class CustomEnumInt32 : int {};
enum class CustomEnumUInt32 : unsigned int {};
enum class CustomEnumInt64 : long int {};
enum class CustomEnumUInt64 : unsigned long int {};

struct CustomStruct {
   float a = 0.0;
   std::vector<float> v1;
   std::vector<std::vector<float>> v2;
   std::string s;
   std::byte b{0};

   bool operator<(const CustomStruct &c) const { return a < c.a && v1 < c.v1 && v2 < c.v2 && s < c.s; }

   bool operator==(const CustomStruct &c) const { return a == c.a && v1 == c.v1 && v2 == c.v2 && s == c.s; }
};

template <>
struct std::hash<CustomStruct> {
   std::size_t operator()(const CustomStruct &c) const noexcept { return std::hash<float>{}(c.a); }
};

struct DerivedA : public CustomStruct {
   std::vector<float> a_v;
   std::string a_s;
};

struct DerivedA2 : public CustomStruct {
   float a2_f{};
};

struct DerivedB : public DerivedA {
   float b_f1 = 0.0;
   float b_f2 = 0.0; //!
   std::string b_s;
};

struct DerivedC : public DerivedA, public DerivedA2 {
   int c_i{};
   DerivedA2 c_a2;
};

struct StructWithArrays {
   unsigned char c[4];
   float f[2];
   int i[2][1];
};

struct EmptyStruct {};
struct alignas(std::uint64_t) TestEBO : public EmptyStruct {
   std::uint64_t u64;
};

template <typename T>
class EdmWrapper {
public:
   bool fIsPresent = true;
   T fMember;
};

class IOConstructor {
public:
   IOConstructor() = delete;
   IOConstructor(TRootIOCtor *) {};

   int a = 7;
};

class LowPrecisionFloats {
public:
   double a = 0.0;
   Double32_t b = 1.0;
   Double32_t c[2] = {2.0, 3.0};
};

/// The classes below are based on an excerpt provided by Marcin Nowak (EP-UAT)
///
struct IAuxSetOption {};
namespace SG { typedef uint32_t sgkey_t; }

struct PackedParameters {
   uint8_t m_nbits, m_nmantissa;
   float   m_scale;
   uint8_t m_flags;
   SG::sgkey_t  m_sgkey = 123;
   const uint8_t c_uint = 10;
};

template <class T>
struct PackedContainer : public std::vector<T>, public IAuxSetOption {
   PackedContainer() = default;
   PackedContainer(std::initializer_list<T> l, const PackedParameters& p) : std::vector<T>(l), m_params(p) {}

   PackedParameters m_params;
};

/// class with non-trivial constructor and destructor
struct ComplexStruct {
   static int gNCallConstructor;
   static int gNCallDestructor;

   static int GetNCallConstructor();
   static int GetNCallDestructor();
   static void SetNCallConstructor(int);
   static void SetNCallDestructor(int);

   ComplexStruct();
   ~ComplexStruct();

   int a = 0;
};

/// Classes with enum declarations (see #8901)
struct BaseOfStructWithEnums {
   int E1;
};

struct StructWithEnums : BaseOfStructWithEnums {
   enum { A1, A2 };
   enum DeclE { E1, E2, E42 = 42 };
   enum class DeclEC { E1, E2, E42 = 137 };
   int a = E42;
   int b = static_cast<int>(DeclEC::E42);
   CustomEnum e = kCustomEnumVal;
};

/// A class that behaves as a collection accessed through the `TVirtualCollectionProxy` interface
template <typename T>
struct StructUsingCollectionProxy {
   using ValueType = T;
   std::vector<T> v; //! do not accidentally store via RClassField
};

/// Classes to exercise field traits
struct TrivialTraitsBase {
   int a;
};

struct TrivialTraits : TrivialTraitsBase {
   float b;
};

struct TransientTraits : TrivialTraitsBase {
   float b; //! transient member
};

struct VariantTraitsBase {
   std::variant<int, float> a;
};

struct VariantTraits : VariantTraitsBase {
};

struct StringTraits : VariantTraitsBase {
   std::string s;
};

struct ConstructorTraits : TrivialTraitsBase {
   ConstructorTraits() {}
};

struct DestructorTraits : TrivialTraitsBase {
   ~DestructorTraits() {}
};

struct StructWithIORulesBase {
   float a;
   float b; //! transient member
};

struct StructWithTransientString {
   char chars[4];
   std::string str; //! transient member
};

struct StructWithIORules : StructWithIORulesBase {
   StructWithTransientString s;
   float c = 0.0f; //! transient member

   StructWithIORules() = default;
   StructWithIORules(float _a, char _c[4]) : StructWithIORulesBase{_a, 0.0f}, s{{_c[0], _c[1], _c[2], _c[3]}, {}} {}
};

#endif
