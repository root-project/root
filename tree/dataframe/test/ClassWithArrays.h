#ifndef DATASOURCE_NTUPLE_TEST_CLASS_WITH_ARRAYS
#define DATASOURCE_NTUPLE_TEST_CLASS_WITH_ARRAYS

#include <array>

#include <Rtypes.h>
#include <ROOT/RVec.hxx>

class ClassWithArrays {
public:
   virtual ~ClassWithArrays() {} // to make dictionary generation happy

   std::array<float, 3> fArr{};
   std::array<ROOT::RVecF, 3> fArrRVec{};
   ROOT::RVec<std::array<float, 3>> fRVecArr{};

   ClassWithArrays() {}

   ClassWithArrays(std::array<float, 3> &&arr, std::array<ROOT::RVecF, 3> &&arrRvec,
                   ROOT::RVec<std::array<float, 3>> &&rVecArr)
      : fArr(std::move(arr)), fArrRVec(std::move(arrRvec)), fRVecArr(std::move(rVecArr))
   {
   }

   ClassDef(ClassWithArrays, 2)
};

#endif // DATASOURCE_NTUPLE_TEST_CLASS_WITH_ARRAYS
