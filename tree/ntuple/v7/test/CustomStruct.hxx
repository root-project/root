#ifndef ROOT7_RNTuple_Test_CustomStruct
#define ROOT7_RNTuple_Test_CustomStruct

#include <string>
#include <vector>

/**
 * Used to test serialization and deserialization of classes in RNTuple with TClass
 */
struct CustomStruct {
   float a = 0.0;
   std::vector<float> v1;
   std::vector<std::vector<float>> v2;
   std::string s;
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

struct EmptyBase {
};
struct alignas(std::uint64_t) TestEBO : public EmptyBase {
   std::uint64_t u64;
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
};

#endif
