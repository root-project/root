#include"Math/Util.h"
#include <vector>

int KahanTest()
{
   std::vector<double> numbers = {0.01, 0.001, 0.0001, 0.000001, 0.00000000001};
   ROOT::Math::KahanSum<double> k;
   k.Add(numbers);
   auto result = ROOT::Math::KahanSum<double>::Accumulate(numbers);
   if(k.Result()!=result)
      return 1;
   return 0;
}

int main(){
      return KahanTest();
}