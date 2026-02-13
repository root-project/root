#ifndef ROOT_DATAFRAME_TEST_ARRRAYCOMBINATIONS
#define ROOT_DATAFRAME_TEST_ARRRAYCOMBINATIONS

#include <array>
#include <vector>

#include <Rtypes.h>

struct ClassWithSequenceContainers {
   unsigned int fObjIndex{};
   std::array<float, 3> fArrFl{};
   std::array<std::array<float, 3>, 3> fArrArrFl{};
   std::array<std::vector<float>, 3> fArrVecFl{};

   std::vector<float> fVecFl{};
   std::vector<std::array<float, 3>> fVecArrFl{}; //! Not supported for TTree: could not find the real data member
                                                  //! '_M_elems[3]' when constructing the branch 'fVecArrFl'
   std::vector<std::vector<float>> fVecVecFl{};

   // For ROOT I/O
   ClassWithSequenceContainers() = default;

   ClassWithSequenceContainers(unsigned int objIndex, std::array<float, 3> a1, std::array<std::array<float, 3>, 3> a2,
                               std::array<std::vector<float>, 3> a3, std::vector<float> a4,
                               std::vector<std::array<float, 3>> a5, std::vector<std::vector<float>> a6)
      : fObjIndex(objIndex),
        fArrFl(std::move(a1)),
        fArrArrFl(std::move(a2)),
        fArrVecFl(std::move(a3)),
        fVecFl(std::move(a4)),
        fVecArrFl(std::move(a5)),
        fVecVecFl(std::move(a6))
   {
   }

   ClassDefNV(ClassWithSequenceContainers, 1)
};

#endif
