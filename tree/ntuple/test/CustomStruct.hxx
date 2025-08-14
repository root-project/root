#ifndef ROOT_RNTuple_Test_CustomStruct
#define ROOT_RNTuple_Test_CustomStruct

#include <RtypesCore.h> // for Double32_t
#include <TObject.h>
#include <TRootIOCtor.h>
#include <TVirtualCollectionProxy.h>

#include <chrono>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
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
   template <typename T>
   using MyVec = std::vector<T>;

   float a = 0.0;
   std::vector<float> v1;
   std::vector<std::vector<float>> v2;
   std::string s;
   std::byte b{};

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

struct DerivedWithTypedef : public CustomStruct {
   MyVec<int> m;
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

template <typename T>
struct EdmHashTrait {
   using value_type = T;
};

template <int I>
class EdmHash {
public:
   typedef std::string value_type;
   value_type fHash;

   template <typename T>
   using value_typeT = typename EdmHashTrait<T>::value_type;
   value_typeT<value_type> fHash2;
};

template <typename FirstT, typename SecondT = double>
class DataVector {
public:
   class Inner {
      FirstT fFirst;
      SecondT fSecond;
   };

   template <typename FirstU, typename SecondU = double>
   class Nested {
      FirstU fFirst;
      SecondU fSecond;
   };

   FirstT fFirst;
   SecondT fSecond;
};

template <typename T1, typename T2, typename T3, typename T4>
class InnerCV {};

template <long long TLL, unsigned long long TULL>
class IntegerTemplates {};

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
   float cDerived = 0.0f;    //! should become 2*c after rules for c applied
   float checksumA = 0.0f;   //! transient member, edited by checksum based rule
   float checksumB = 137.0f; //! transient member, skipped by checksum based rule due to checksum mismatch

   StructWithIORules() = default;
   StructWithIORules(float _a, char _c[4]) : StructWithIORulesBase{_a, 0.0f}, s{{_c[0], _c[1], _c[2], _c[3]}, {}} {}
};

struct OldCoordinates {
   float fOldX;
   float fOldY;
};

struct CoordinatesWithIORules {
   float fX;
   float fY;
   float fR;   //!
   float fPhi; //!
};

struct LowPrecisionFloatWithIORules {
   float fFoo;
   float fLast8BitsZero; // the I/O rule will "randomize" the last 8 bits
};

template <typename T>
struct OldName {
   T fValue;
};

template <typename T>
struct NewName {
   T fValue;
};

struct SourceStruct {
   int fValue;
   int fTransient; //!
   SourceStruct()
   {
      fValue = 17;
      fTransient = 23;
   }
};

struct StructWithSourceStruct {
   SourceStruct fSource;
   int fTransient = 0; //!
};

struct Cyclic {
   std::vector<Cyclic> fMember;
};

// Test cyclic collection proxy: we set up this class such that it is its own collection proxy inner class.
// This does not actually need to be a working collection proxy.
struct CyclicCollectionProxy : public TVirtualCollectionProxy {
   // The following three functions are required by RProxiedCollectionField
   static void Func_CreateIterators(void *, void **, void **, TVirtualCollectionProxy *) {}
   static void *Func_Next(void *, const void *) { return nullptr; }
   static void Func_DeleteTwoIterators(void *, void *) {}

public:
   CyclicCollectionProxy();
   TVirtualCollectionProxy *Generate() const final { return new CyclicCollectionProxy(); }
   Int_t GetCollectionType() const final { return 0; }
   ULong_t GetIncrement() const final { return 0; }
   UInt_t Sizeof() const final { return 0; }
   bool HasPointers() const final { return false; }
   TClass *GetValueClass() const final;
   EDataType GetType() const final { return EDataType::kOther_t; }
   void PushProxy(void *) final {}
   void PopProxy() final {}
   void *At(UInt_t) final { return nullptr; }
   void Clear(const char * = "") final {}
   UInt_t Size() const final { return 0; }
   void *Allocate(UInt_t, bool) final { return nullptr; }
   void Commit(void *) final {}
   void Insert(const void *, void *, size_t) final {}
   TStreamerInfoActions::TActionSequence *GetConversionReadMemberWiseActions(TClass *, Int_t) final { return nullptr; }
   TStreamerInfoActions::TActionSequence *GetReadMemberWiseActions(Int_t) final { return nullptr; }
   TStreamerInfoActions::TActionSequence *GetWriteMemberWiseActions() final { return nullptr; }
   CreateIterators_t GetFunctionCreateIterators(bool = true) final { return &Func_CreateIterators; }
   CopyIterator_t GetFunctionCopyIterator(bool = true) final { return nullptr; }
   Next_t GetFunctionNext(bool = true) final { return &Func_Next; }
   DeleteIterator_t GetFunctionDeleteIterator(bool = true) final { return nullptr; }
   DeleteTwoIterators_t GetFunctionDeleteTwoIterators(bool = true) final { return &Func_DeleteTwoIterators; }
};

struct Unsupported {
   float a;
   std::chrono::time_point<std::chrono::system_clock> timestamp;
   float b;
   std::random_device rd;
   float z;
};

struct BaseA {
   float a = 0.0;
};

struct DiamondVirtualB : virtual public BaseA {
   float b = 0.0;
};

struct DiamondVirtualC : virtual public BaseA {
   float c = 0.0;
};

struct DiamondVirtualD : public DiamondVirtualB, public DiamondVirtualC {
   float d = 0.0;
};

struct DuplicateBaseB : public BaseA {
   float b = 0.0;
};

struct DuplicateBaseC : public BaseA {
   float c = 0.0;
};

struct DuplicateBaseD : public DuplicateBaseB, public DuplicateBaseC {
   float d = 0.0;
};

class Left {
public:
   float x = 1.0;
   virtual ~Left() = default;
   ClassDef(Left, 1)
};

class DerivedFromLeftAndTObject : public Left, public TObject {
public:
   virtual ~DerivedFromLeftAndTObject() = default;
   ClassDefOverride(DerivedFromLeftAndTObject, 1)
};

struct ThrowForVariant {
   ThrowForVariant() = default;
   ThrowForVariant(const ThrowForVariant &) { throw std::runtime_error("copy ctor"); }
   ThrowForVariant &operator=(const ThrowForVariant &) = default;
};

struct RelativelyLargeStruct {
   char fDummy[250];
};

// Adjusted from
// https://root-forum.cern.ch/t/manual-schema-evolution-with-i-o-rules-and-branches-containing-vector-t/64026

namespace v1 {
struct Vector3D {
   float fX = 0.;
   float fY = 0.;
   float fZ = 0.;
};

struct ExampleMC {
   float fEnergy = 0.;
   v1::Vector3D fSpin;
};
} // namespace v1

namespace v2 {
struct ExampleMC {
   float fEnergy = 0.;
   float fHelicity = 0.;
};
} // namespace v2

#endif
