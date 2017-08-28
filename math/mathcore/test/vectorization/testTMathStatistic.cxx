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
* Common data and set up for testing vectorized functions from TMath
*/
class TVectorizedMathTest : public ::testing::Test {
protected:
   TVectorizedMathTest()
   {
      // Randomize input data and parameters
      TRandom1 rndmzr;
      rndmzr.RndmArray(fInputSize * vecCore::VectorSize<ROOT::Double_v>(), fLinearInput);
      rndmzr.RndmArray(fInputSize * vecCore::VectorSize<ROOT::Double_v>(), fLinearProbabilities);
      rndmzr.RndmArray(fInputSize, fMean);
      rndmzr.RndmArray(fInputSize, fSigma);

      // Set -100 < mean < 100 and 0 < sigma < 200
      for (size_t i = 0; i < fInputSize; i++) {
         fMean[i] = fMean[i] * 200 - 100;
         fSigma[i] *= 200;
      }

      // Copy input linear data to the vectorized array
      for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
         for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
            vecCore::Set(fVectorInput[caseIdx], vecIdx,
                         fLinearInput[vecCore::VectorSize<ROOT::Double_v>() * caseIdx + vecIdx]);
            vecCore::Set(fVectorProbabilities[caseIdx], vecIdx,
                         fLinearProbabilities[vecCore::VectorSize<ROOT::Double_v>() * caseIdx + vecIdx]);
         }
      }
   }

   // Data available to all tests in this suite

   static const int fInputSize = 100000;

   // Probabilities (values in (0,1)) input
   ROOT::Double_v fVectorProbabilities[fInputSize];
   Double_t fLinearProbabilities[fInputSize * vecCore::VectorSize<ROOT::Double_v>()];

   // Vectorized and linear input
   ROOT::Double_v fVectorInput[fInputSize];
   Double_t fLinearInput[fInputSize * vecCore::VectorSize<ROOT::Double_v>()];

   // Vectorized and linear output
   ROOT::Double_v fVectorOutput[fInputSize];
   Double_t fLinearOutput[fInputSize * vecCore::VectorSize<ROOT::Double_v>()];

   // Parameters vector
   Double_t fMean[fInputSize];
   Double_t fSigma[fInputSize];
};

// Check that the scalar and vectorial versions of TMath::Gaus produce the same results
TEST_F(TVectorizedMathTest, VectorizedGaus)
{
   // Compute the vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      fVectorOutput[caseIdx] = TMath::Gaus(fVectorInput[caseIdx], fMean[caseIdx], fSigma[caseIdx]);
   }

   // Compute the linear output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         fLinearOutput[linear_index] = TMath::Gaus(fLinearInput[linear_index], fMean[caseIdx], fSigma[caseIdx]);
      }
   }

   // Compare linear and vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         EXPECT_NEAR(fLinearOutput[linear_index], vecCore::Get<ROOT::Double_v>(fVectorOutput[caseIdx], vecIdx), 1e-15);
      }
   }
}

// Check that the scalar and vectorial versions of TMath::BretiWigner produce the same results
TEST_F(TVectorizedMathTest, VectorizedBreitWigner)
{
   // Compute the vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      fVectorOutput[caseIdx] = TMath::BreitWigner(fVectorInput[caseIdx], fMean[caseIdx], fSigma[caseIdx]);
   }

   // Compute the linear output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         fLinearOutput[linear_index] = TMath::BreitWigner(fLinearInput[linear_index], fMean[caseIdx], fSigma[caseIdx]);
      }
   }

   // Compare linear and vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         EXPECT_NEAR(fLinearOutput[linear_index], vecCore::Get<ROOT::Double_v>(fVectorOutput[caseIdx], vecIdx), 1e-15);
      }
   }
}

// Check that the scalar and vectorial versions of TMath::CauchyDist produce the same results
TEST_F(TVectorizedMathTest, VectorizedCauchyDist)
{
   // Compute the vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      fVectorOutput[caseIdx] = TMath::CauchyDist(fVectorInput[caseIdx], fMean[caseIdx], fSigma[caseIdx]);
   }

   // Compute the linear output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         fLinearOutput[linear_index] = TMath::CauchyDist(fLinearInput[linear_index], fMean[caseIdx], fSigma[caseIdx]);
      }
   }

   // Compare linear and vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         EXPECT_NEAR(fLinearOutput[linear_index], vecCore::Get<ROOT::Double_v>(fVectorOutput[caseIdx], vecIdx), 1e-15);
      }
   }
}

// Check that the scalar and vectorial versions of TMath::LaplaceDist produce the same results
TEST_F(TVectorizedMathTest, VectorizedLaplaceDist)
{
   // Compute the vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      fVectorOutput[caseIdx] = TMath::LaplaceDist(fVectorInput[caseIdx], fMean[caseIdx], fSigma[caseIdx]);
   }

   // Compute the linear output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         fLinearOutput[linear_index] = TMath::LaplaceDist(fLinearInput[linear_index], fMean[caseIdx], fSigma[caseIdx]);
      }
   }

   // Compare linear and vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         EXPECT_NEAR(fLinearOutput[linear_index], vecCore::Get<ROOT::Double_v>(fVectorOutput[caseIdx], vecIdx), 1e-15);
      }
   }
}

// Check that the scalar and vectorial versions of TMath::LaplaceDistI produce the same results
TEST_F(TVectorizedMathTest, VectorizedLaplaceDistI)
{
   // Compute the vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      fVectorOutput[caseIdx] = TMath::LaplaceDistI(fVectorInput[caseIdx], fMean[caseIdx], fSigma[caseIdx]);
   }

   // Compute the linear output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         fLinearOutput[linear_index] = TMath::LaplaceDistI(fLinearInput[linear_index], fMean[caseIdx], fSigma[caseIdx]);
      }
   }

   // Compare linear and vectorized output
   for (size_t caseIdx = 0; caseIdx < fInputSize; caseIdx++) {
      for (size_t vecIdx = 0; vecIdx < vecCore::VectorSize<ROOT::Double_v>(); vecIdx++) {
         size_t linear_index = caseIdx * vecCore::VectorSize<ROOT::Double_v>() + vecIdx;
         EXPECT_NEAR(fLinearOutput[linear_index], vecCore::Get<ROOT::Double_v>(fVectorOutput[caseIdx], vecIdx), 1e-15);
      }
   }
}

#endif // R__HAS_VECCORE
