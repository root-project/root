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

/// The classes below are based on an excerpt provided by Marcin Nowak (EP-UAT)
///
struct IAuxSetOption {};

struct PackedParameters {
   uint8_t m_nbits, m_nmantissa;
   float   m_scale;
   uint8_t m_flags;
};

template <class T>
struct PackedContainer : public std::vector<T>, public IAuxSetOption {
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
