#include "TMath.h"
#include "TMathVectorized.h"

#include "Math/Random.h"
#include "TRandom1.h"
#include "TStopwatch.h"

#include "gtest/gtest.h"

#include <iostream>
#include <random>

#ifdef R__HAS_VECCORE

#include <VecCore/VecCore>

/**
 * Common data and set up for testing vectorized functions from TMathBase
 */
class TVectorizedMathBaseTest : public ::testing::Test {
protected:
   TVectorizedMathBaseTest()
   {
      // Randomize linear input data in ]0, 1]
      TRandom1 rndmzr;
      rndmzr.RndmArray(fInputSize * vecCore::VectorSize<ROOT::Double_v>(), fLinearInputA);
      rndmzr.RndmArray(fInputSize * vecCore::VectorSize<ROOT::Double_v>(), fLinearInputB);

      // Transform input linear data to ]-1000, 1000] and copy to the vectorized array
      for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
         for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
            size_t linear_index = vecCore::VectorSize<ROOT::Double_v>() * caseIdx + vecIdx;

            // Transform data ]0, 1] -> ]-1000, 1000]
            fLinearInputA[linear_index] *= 2000;
            fLinearInputA[linear_index] -= 1000;

            vecCore::Set(fVectorInputA[caseIdx], vecIdx, fLinearInputA[linear_index]);
            vecCore::Set(fVectorInputB[caseIdx], vecIdx, fLinearInputB[linear_index]);
         }
      }
   }

   // Data available to all tests in this suite

   static const int fInputSize = 1000;

   // Vectorized and linear input
   ROOT::Double_v fVectorInputA[fInputSize];
   ROOT::Double_v fVectorInputB[fInputSize];
   Double_t fLinearInputA[fInputSize * vecCore::VectorSize<ROOT::Double_v>()];
   Double_t fLinearInputB[fInputSize * vecCore::VectorSize<ROOT::Double_v>()];

   // Vectorized and linear output
   ROOT::Double_v fVectorOutput[fInputSize];
   Double_t fLinearOutput[fInputSize * vecCore::VectorSize<ROOT::Double_v>()];
};

// Test scalar and vectorial versions of TMath::Min produce the same results
TEST_F(TVectorizedMathBaseTest, VectorizedMin)
{
   // Compute output with vectorized function
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      fVectorOutput[caseIdx] = TMath::Min(fVectorInputA[caseIdx], fVectorInputB[caseIdx]);
   }

   // Compute output with linear function
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t idx = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         fLinearOutput[idx] = TMath::Min(fLinearInputA[idx], fLinearInputB[idx]);
      }
   }

   // Compare linear and vector output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         EXPECT_EQ(fLinearOutput[caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx],
                   vecCore::Get(fVectorOutput[caseIdx], vecIdx));
      }
   }
}

#endif // R__HAS_VECCORE
