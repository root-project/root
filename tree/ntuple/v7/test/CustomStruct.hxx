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
