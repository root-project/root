#include "Math/Util.h"
#include <vector>

int KahanTest()
{
   std::vector<double> numbers = {0.01, 0.001, 0.0001, 0.000001, 0.00000000001};
   ROOT::Math::KahanSum<double> k;
   k.Add(numbers.begin(), numbers.end());
   auto result = ROOT::Math::KahanSum<double>::Accumulate(numbers.begin(), numbers.end());
   if (k.Result() != result) return 1;

   ROOT::Math::KahanSum<double> k2;
   ROOT::Math::KahanSum<double> k3(1);
   k2.Add(1);
   k2.Add(numbers.begin(), numbers.end());
   k3.Add(numbers.begin(), numbers.end());
   if (k2.Result() != k3.Result()) return 2;

   return 0;
}

int main()
{
   return KahanTest();
}