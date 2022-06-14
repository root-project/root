#include <cstdio>
#include <VecCore/VecCore>

int main(int, char **)
{
   printf("VecCore available backends: Scalar ");
#ifdef VECCORE_ENABLE_UMESIMD
   printf("UME::SIMD ");
#endif
#ifdef VECCORE_ENABLE_VC
   printf("Vc");
#endif
   printf("\n");
   return 0;
}
