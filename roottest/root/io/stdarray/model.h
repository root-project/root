#include <array>

namespace edm0 {

class B{};

class A{
private:
   std::array<int,3> a0 {{3,6,9}};
   int a1[3] = {3,6,9};

   std::array<std::array<int,3>,3> a2 {{ {{1,2,3}},{{1,2,3}},{{1,2,3}} }};
   int a3[3][3] = {{1,2,3},{1,2,3},{1,2,3}};

   std::array<B,42> a4;
   B a5[42];

   std::array<float,3> a6 {{3,6,9}};
   float a7[3] = {3,6,9};

   std::array<std::array<float,3>,3> a8 {{ {{1,2,3}},{{1,2,3}},{{1,2,3}} }};
   float a9[3][3] = {{1,2,3},{1,2,3},{1,2,3}};
};
}
