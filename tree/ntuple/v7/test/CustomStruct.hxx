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
   CustomStruct(float w = 0.0f, std::vector<float> x = {0.0f},
      std::vector<std::vector<float>> y = {{0.0f}}, std::string z = ""):
      a(w), v1(x), v2(y), s(z) { }
};
