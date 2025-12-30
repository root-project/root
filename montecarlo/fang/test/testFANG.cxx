// @(#)root/physics:$Id$
// Author: Arik Kreisel, Itay Horin

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// \file testFANG.cxx
/// \ingroup Physics
/// \brief Unit tests for FANG (Focused Angular N-body event Generator)
/// \authors Arik Kreisel, Itay Horin
///
/// This file contains gtest unit tests for the FANG phase space generator:
/// 1. Full phase space calculation validation
/// 2. Partial phase space with detector constraints
/// 3. Elastic ep scattering differential cross section vs Rosenbluth formula
///
/// Reference: "Focused Angular N-Body Event Generator (FANG)" paper
/// https://arxiv.org/abs/2509.11105
////////////////////////////////////////////////////////////////////////////////

#include "FANG.h"

#include "gtest/gtest.h"
#include "TError.h"
#include "TRandom3.h"
#include "TMath.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include <vector>
#include <cmath>

using namespace FANG;

////////////////////////////////////////////////////////////////////////////////
/// Test fixture for FANG tests
////////////////////////////////////////////////////////////////////////////////
class FANGTest : public ::testing::Test {
protected:
   TRandom3 rng;

   void SetUp() override
   {
      // Set random seed for reproducibility in tests
      rng.SetSeed(12345);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// \brief Test CalcKMFactor with known values
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, CalcKMFactor_KnownValues)
{
   // F(0,0) = sqrt((1-0-0)^2 - 4*0*0) = 1
   EXPECT_DOUBLE_EQ(CalcKMFactor(0.0, 0.0), 1.0);

   // F(0.25, 0.25) = sqrt((1-0.5)^2 - 4*0.0625) = sqrt(0.25 - 0.25) = 0
   EXPECT_NEAR(CalcKMFactor(0.25, 0.25), 0.0, 1e-10);

   // F(0.1, 0.1) = sqrt((0.8)^2 - 0.04) = sqrt(0.64 - 0.04) = sqrt(0.6)
   EXPECT_NEAR(CalcKMFactor(0.1, 0.1), std::sqrt(0.6), 1e-10);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test TwoBody decay conserves 4-momentum
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, TwoBody_MomentumConservation)
{
   ROOT::Math::PxPyPzMVector S(1.0, 2.0, 3.0, 5.0);
   Double_t m1 = 1.0;
   Double_t m2 = 1.5;
   ROOT::Math::PxPyPzMVector p1, p2;

   TwoBody(S, m1, m2, p1, p2, &rng);

   // Check 4-momentum conservation
   ROOT::Math::PxPyPzMVector pSum = p1 + p2;
   EXPECT_NEAR(pSum.Px(), S.Px(), 1e-10);
   EXPECT_NEAR(pSum.Py(), S.Py(), 1e-10);
   EXPECT_NEAR(pSum.Pz(), S.Pz(), 1e-10);
   EXPECT_NEAR(pSum.E(), S.E(), 1e-10);

   // Check masses are correct
   EXPECT_NEAR(p1.M(), m1, 1e-10);
   EXPECT_NEAR(p2.M(), m2, 1e-10);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test TwoBody decay produces physical momenta
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, TwoBody_PhysicalMomenta)
{
   ROOT::Math::PxPyPzMVector S(0.0, 0.0, 5.0, 10.0);
   Double_t m1 = 2.0;
   Double_t m2 = 3.0;
   ROOT::Math::PxPyPzMVector p1, p2;

   TwoBody(S, m1, m2, p1, p2, &rng);

   // Check energies are positive and >= mass
   EXPECT_GE(p1.E(), m1);
   EXPECT_GE(p2.E(), m2);

   // Check 3-momentum magnitudes are positive
   EXPECT_GE(p1.P(), 0.0);
   EXPECT_GE(p2.P(), 0.0);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test GenFANG full phase space with known reference value
///
/// Uses P(0,0,5,M=12) decaying to 5 particles of mass 1 each.
/// Reference value from FANG paper Table I: 26628.1 Ã‚Â± 3.0
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, FullPhaseSpace_ReferenceValue)
{
   const Int_t kNBody = 5;
   Double_t masses[kNBody] = {1.0, 1.0, 1.0, 1.0, 1.0};
   ROOT::Math::PxPyPzMVector pTotal(0, 0, 5, 12);  // Note: E=13 as in paper

   std::vector<ROOT::Math::XYZVector> v3Det;
   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   Double_t omega0[1];
   Double_t shape0[1];

   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;
   Int_t nEvents = 0;
   const Int_t nLoop = 500000;

   for (Int_t k = 0; k < nLoop; k++) {
      vecVecP.clear();
      vecWi.clear();
      Int_t status = GenFANG(kNBody, pTotal, masses, omega0, shape0, v3Det, vecVecP, vecWi, &rng);
      if (!status) continue;

      for (size_t i = 0; i < vecVecP.size(); i++) {
         Double_t weight = vecWi[i];
         nEvents++;
         sumW += weight;
         sumW2 += weight * weight;
      }
   }

   Double_t phaseSpace = sumW / nEvents;
   Double_t error = TMath::Sqrt(sumW2) / nEvents;

   // Reference value from paper: 26628.1 Ã‚Â± 3.0
   // Allow 0.5% tolerance for Monte Carlo fluctuations
   Double_t expectedValue = 26628.1;
   Double_t tolerance = 0.005 * expectedValue;

   EXPECT_NEAR(phaseSpace, expectedValue, tolerance)
      << "Phase space = " << phaseSpace << " +/- " << error
      << ", expected = " << expectedValue;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test GenFANG partial phase space with detector constraints
///
/// 
/// and that constrained particles are within specified solid angles.
/// test to Reference value 4.7151 
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, PartialPhaseSpace_Constraints)
{
   const Int_t kNBody = 5;
   const Int_t kNDet = 3;
   Double_t masses[kNBody] = {1.0, 1.0, 1.0, 1.0, 1.0};
   ROOT::Math::PxPyPzMVector pTotal(0, 0, 5, 12);

   std::vector<ROOT::Math::XYZVector> v3Det;
   Double_t omega[kNDet];
   Double_t shape[kNDet];

   // Detector 1: Circle at (0, 0, 0.5), radius 0.2
   ROOT::Math::XYZVector v3_1(0.0, 0.0, 0.5);
   v3Det.push_back(v3_1);
   Double_t radius1 = TMath::Sqrt(v3_1.Mag2() + 0.2 * 0.2);
   omega[0] = kTwoPi * radius1 * (radius1 - v3_1.R());
   shape[0] = 0.0;  // Circle

   // Detector 2: Circle at (0.5, 0, 0), radius 0.3
   ROOT::Math::XYZVector v3_2(0.5, 0.0, 0.0);
   v3Det.push_back(v3_2);
   Double_t radius2 = TMath::Sqrt(v3_2.Mag2() + 0.3 * 0.3);
   omega[1] = kTwoPi * radius2 * (radius2 - v3_2.R());
   shape[1] = 0.0;  // Circle

   // Detector 3: Strip at (0, 0.5, 0)
   ROOT::Math::XYZVector v3_3(0.0, 0.5, 0.0);
   v3Det.push_back(v3_3);
   omega[2] = 1.2 * kPi;
   shape[2] = 0.4;  // Strip

   Double_t totalOmega = omega[0] * omega[1] * omega[2];

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;
   Int_t nEvents = 0;
   const Int_t nLoop = 1000000;

   for (Int_t k = 0; k < nLoop; k++) {
      vecVecP.clear();
      vecWi.clear();
      Int_t status = GenFANG(kNBody, pTotal, masses, omega, shape, v3Det, vecVecP, vecWi, &rng);
      if (!status) continue;

      for (size_t i = 0; i < vecVecP.size(); i++) {
         std::vector<ROOT::Math::PxPyPzMVector> vecP = vecVecP[i];
         Double_t weight = vecWi[i];
         nEvents++;
         sumW += weight;
         sumW2 += weight * weight;

         // Verify momentum conservation
         ROOT::Math::PxPyPzMVector pSum;
         for (size_t j = 0; j < vecP.size(); j++) {
            pSum = pSum + vecP[j];
         }
         EXPECT_NEAR(pSum.Px(), pTotal.Px(), 1e-8);
         EXPECT_NEAR(pSum.Py(), pTotal.Py(), 1e-8);
         EXPECT_NEAR(pSum.Pz(), pTotal.Pz(), 1e-8);
         EXPECT_NEAR(pSum.E(), pTotal.E(), 1e-6);
      }
   }
    Double_t partialphaseSpace=totalOmega * sumW / nEvents;
    Double_t perror= totalOmega * TMath::Sqrt(sumW2) / nEvents;
   
   // Reference value from paper: 26628.1 Ã‚Â± 3.0
   // Allow 5% tolerance for Monte Carlo fluctuations
   Double_t pexpectedValue = 4.764 ;
   Double_t ptolerance = 0.05 * pexpectedValue;

   EXPECT_NEAR(partialphaseSpace, pexpectedValue, ptolerance)
      << "Partial Phase space = " << partialphaseSpace << " +/- " << perror
      << ", expected = " << pexpectedValue;

}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test GenFANG two-body constrained decay
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, TwoBody_Constrained)
{
   const Int_t kNBody = 2;
   Double_t masses[kNBody] = {1.0, 2.0};
   ROOT::Math::PxPyPzMVector pTotal(0, 0, 3, 6);

   std::vector<ROOT::Math::XYZVector> v3Det;
   ROOT::Math::XYZVector v3(0.0, 0.0, 1.0);  // Constrain to z-axis
   v3Det.push_back(v3);

   Double_t omega[1] = {0.0};
   Double_t shape[1] = {kModePoint};  // Point generation

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   Int_t status = GenFANG(kNBody, pTotal, masses, omega, shape, v3Det, vecVecP, vecWi, &rng);
   EXPECT_EQ(status, 1) << "GenFANG should succeed";
   EXPECT_GE(vecVecP.size(), 1u) << "Should have at least one solution";

   for (size_t i = 0; i < vecVecP.size(); i++) {
      std::vector<ROOT::Math::PxPyPzMVector> vecP = vecVecP[i];
      EXPECT_EQ(vecP.size(), 2u) << "Should have 2 particles";

      // Check masses
      EXPECT_NEAR(vecP[0].M(), masses[0], 1e-10);
      EXPECT_NEAR(vecP[1].M(), masses[1], 1e-10);

      // Check momentum conservation
      ROOT::Math::PxPyPzMVector pSum = vecP[0] + vecP[1];
      EXPECT_NEAR(pSum.Px(), pTotal.Px(), 1e-10);
      EXPECT_NEAR(pSum.Py(), pTotal.Py(), 1e-10);
      EXPECT_NEAR(pSum.Pz(), pTotal.Pz(), 1e-10);
      EXPECT_NEAR(pSum.E(), pTotal.E(), 1e-8);

      // First particle should be along z-axis (within numerical precision)
      if (vecP[0].P() > 1e-10) {
         Double_t cosTheta = vecP[0].Pz() / vecP[0].P();
         EXPECT_NEAR(std::abs(cosTheta), 1.0, 1e-10)
            << "Constrained particle should be along z-axis";
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Rosenbluth Cross Section Helper Functions
////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate Rosenbluth cross section for elastic ep scattering
/// \param[in] cosTheta cos(theta) in lab frame
/// \param[in] kineticE electron kinetic energy [GeV]
/// \return Differential cross section dsigma/dOmega [GeV^-2]
////////////////////////////////////////////////////////////////////////////////
Double_t RosenbluthCrossSection(Double_t cosTheta, Double_t kineticE)
{
   Double_t sigma = 0.0;
   const Double_t alpha = 1.0 / 137.0;
   const Double_t massProton = 0.938272029;
   const Double_t massElectron = 0.000511;

   ROOT::Math::XYZVector vDir(TMath::Sqrt(1.0 - cosTheta * cosTheta), 0.0, cosTheta);

   ROOT::Math::PxPyPzMVector pProton(0.0, 0.0, 0.0, massProton);
   Double_t gamma = kineticE / massElectron + 1.0;
   Double_t beta = TMath::Sqrt(1.0 - 1.0 / (gamma * gamma));
   ROOT::Math::PxPyPzMVector pElectron(0.0, 0.0, gamma * beta * massElectron, massElectron);

   ROOT::Math::PxPyPzMVector pTotal = pProton + pElectron;

   LongDouble_t massCM = pTotal.M();
   LongDouble_t energyCM = pTotal.E();
   LongDouble_t momCM = pTotal.P();
   LongDouble_t energyCM3 = (massCM * massCM - massProton * massProton +
                             massElectron * massElectron) / (2.0 * massCM);

   LongDouble_t aa = momCM * momCM * cosTheta * cosTheta - energyCM * energyCM;
   LongDouble_t bb = 2.0 * momCM * cosTheta * energyCM3 * massCM;
   LongDouble_t cc = energyCM3 * massCM * energyCM3 * massCM -
                     massElectron * massElectron * energyCM * energyCM;

   if (bb * bb - 4.0 * aa * cc < 0.0) {
      return 0.0;
   }

   LongDouble_t momLAB = (-bb + TMath::Sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa);
   if (momLAB > 0.0) {
      ROOT::Math::PxPyPzMVector pElectronOut(momLAB * vDir.X(), momLAB * vDir.Y(),
                                              momLAB * vDir.Z(), massElectron);
      ROOT::Math::PxPyPzMVector pMomentumTransfer = pElectronOut - pElectron;
      Double_t qSquared = -pMomentumTransfer.M2();
      Double_t formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) *
                               (1.0 + qSquared / kDipoleMassSq));
      Double_t formGM = kProtonMagneticMoment * formGE;
      Double_t tau = qSquared / (4.0 * massProton * massProton);
      Double_t mottXS = alpha * alpha /
                        (pElectron.E() * pElectron.E() * (1.0 - cosTheta) * (1.0 - cosTheta)) *
                        pElectronOut.E() / pElectron.E() * (1.0 + cosTheta) / 2.0;
      sigma = mottXS * ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) +
                        2.0 * tau * formGM * formGM * (1.0 - cosTheta) / (1.0 + cosTheta));
   }

   momLAB = (-bb - TMath::Sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa);
   if (momLAB > 0.0) {
      ROOT::Math::PxPyPzMVector pElectronOut(momLAB * vDir.X(), momLAB * vDir.Y(),
                                              momLAB * vDir.Z(), massElectron);
      ROOT::Math::PxPyPzMVector pMomentumTransfer = pElectronOut - pElectron;
      Double_t qSquared = -pMomentumTransfer.M2();
      Double_t formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) *
                               (1.0 + qSquared / kDipoleMassSq));
      Double_t formGM = kProtonMagneticMoment * formGE;
      Double_t tau = qSquared / (4.0 * massProton * massProton);
      Double_t mottXS = alpha * alpha /
                        (pElectron.E() * pElectron.E() * (1.0 - cosTheta) * (1.0 - cosTheta)) *
                        pElectronOut.E() / pElectron.E() * (1.0 + cosTheta) / 2.0;
      sigma += mottXS * ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) +
                         2.0 * tau * formGM * formGM * (1.0 - cosTheta) / (1.0 + cosTheta));
   }

   return sigma;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate FANG cross section at a specific angle
/// \param[in] cosTheta cos(theta) in lab frame
/// \param[in] kineticE electron kinetic energy [GeV]
/// \param[in] nLoop number of Monte Carlo iterations
/// \param[out] error statistical error estimate
/// \return Differential cross section dsigma/dOmega [GeV^-2]
////////////////////////////////////////////////////////////////////////////////
Double_t FANGCrossSection(Double_t cosTheta, Double_t kineticE, Int_t nLoop, Double_t &error, TRandom3 *rng)
{
   const Int_t kNBody = 2;
   const Double_t massElectron = 0.000511;
   const Double_t massProton = 0.938272029;
   const Double_t alphaQED = 1.0 / 137.0;

   Double_t masses[kNBody] = {massElectron, massProton};

   // Setup kinematics
   ROOT::Math::PxPyPzMVector pTarget(0.0, 0.0, 0.0, massProton);
   Double_t gamma = kineticE / massElectron + 1.0;
   Double_t beta = TMath::Sqrt(1.0 - 1.0 / (gamma * gamma));
   ROOT::Math::PxPyPzMVector pBeam(0.0, 0.0, gamma * beta * massElectron, massElectron);
   ROOT::Math::PxPyPzMVector pTotal = pBeam + pTarget;

   Double_t flux = 1.0 / (16.0 * kPi * kPi *
                          TMath::Sqrt(pBeam.Dot(pTarget) * pBeam.Dot(pTarget) -
                                      massElectron * massElectron * massProton * massProton));

   std::vector<ROOT::Math::XYZVector> v3Det;
   ROOT::Math::XYZVector v3(TMath::Sqrt(1.0 - cosTheta * cosTheta), 0.0, cosTheta);
   v3Det.push_back(v3);

   Double_t omega[1] = {0.0};
   Double_t shape[1] = {kModePoint};

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;
   Int_t nEvents = 0;

   for (Int_t k = 0; k < nLoop; k++) {
      vecVecP.clear();
      vecWi.clear();
      Int_t status = GenFANG(kNBody, pTotal, masses, omega, shape, v3Det, vecVecP, vecWi, rng);
      if (!status) continue;

      for (size_t i = 0; i < vecVecP.size(); i++) {
         std::vector<ROOT::Math::PxPyPzMVector> vecP = vecVecP[i];
         Double_t weight = vecWi[i];

         ROOT::Math::PxPyPzMVector pElectronOut = vecP[0];
         ROOT::Math::PxPyPzMVector pMomTransfer = pBeam - pElectronOut;
         ROOT::Math::PxPyPzMVector pU = pTarget - pElectronOut;
         Double_t qSquared = -pMomTransfer.M2();

         Double_t formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) *
                                  (1.0 + qSquared / kDipoleMassSq));
         Double_t formGM = kProtonMagneticMoment * formGE;
         Double_t tau = qSquared / (4.0 * massProton * massProton);
         Double_t lambda = (pTotal.M2() - pU.M2()) / (4.0 * massProton * massProton);

         Double_t ampSquared = 16.0 * kPi * kPi * alphaQED * alphaQED / (tau * tau) *
                               ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) *
                                (lambda * lambda - tau * tau - tau) +
                                2.0 * tau * tau * formGM * formGM);

         weight *= ampSquared;
         nEvents++;
         sumW += weight;
         sumW2 += weight * weight;
      }
   }

   error = flux * TMath::Sqrt(sumW2) / nEvents;
   return flux * sumW / nEvents;
}

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
/// \brief Test FANG differential cross section against Rosenbluth formula
///
/// Tests elastic ep scattering at 3 GeV for various angles.
/// Skips cos(theta) = Ã‚Â±1 where numerical issues may occur.
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, Rosenbluth_ElasticEP)
{
   const Double_t kineticE = 3.0;  // GeV
   const Int_t nLoop = 50000;

   // Test angles: cos(theta) from -0.8 to 0.8 (skip Ã‚Â±1)
   std::vector<Double_t> testAngles = {-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8};

   for (Double_t cosTheta : testAngles) {
      Double_t rosenbluth = RosenbluthCrossSection(cosTheta, kineticE);
      Double_t fangError;
      Double_t fang = FANGCrossSection(cosTheta, kineticE, nLoop, fangError, &rng);

      // Calculate ratio
      Double_t ratio = fang / rosenbluth;

      // Allow 10% tolerance for Monte Carlo fluctuations at this statistics
      EXPECT_NEAR(ratio, 1.0, 0.10)
         << "cos(theta) = " << cosTheta
         << ": FANG = " << fang << " +/- " << fangError
         << ", Rosenbluth = " << rosenbluth
         << ", ratio = " << ratio;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test FANG cross section precision at a single angle
///
/// Uses higher statistics to verify agreement with Rosenbluth within 5%.
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, Rosenbluth_HighPrecision)
{
   const Double_t kineticE = 3.0;  // GeV
   const Double_t cosTheta = 0.0;  // 90 degree scattering
   const Int_t nLoop = 100000;

   Double_t rosenbluth = RosenbluthCrossSection(cosTheta, kineticE);
   Double_t fangError;
   Double_t fang = FANGCrossSection(cosTheta, kineticE, nLoop, fangError, &rng);

   Double_t ratio = fang / rosenbluth;

   // At high statistics, expect agreement within 5%
   EXPECT_NEAR(ratio, 1.0, 0.05)
      << "High precision test at cos(theta) = 0"
      << ": FANG = " << fang << " +/- " << fangError
      << ", Rosenbluth = " << rosenbluth
      << ", ratio = " << ratio;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test that GenFANG returns 0 for unphysical configurations
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, UnphysicalConfiguration)
{
    // Temporarily suppress error messages since we expect an error condition
   Int_t oldLevel = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kFatal;  // Only show Fatal messages

   const Int_t kNBody = 3;
   Double_t masses[kNBody] = {5.0, 5.0, 5.0};  // Total mass = 15
   ROOT::Math::PxPyPzMVector pTotal(0, 0, 0, 10);  // M = 10 < 15, unphysical

   std::vector<ROOT::Math::XYZVector> v3Det;
   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   Double_t omega[1];
   Double_t shape[1];

   Int_t status = GenFANG(kNBody, pTotal, masses, omega, shape, v3Det, vecVecP, vecWi, &rng);

   gErrorIgnoreLevel = oldLevel;  // Restore previous error level
   EXPECT_EQ(status, 0) << "GenFANG should fail for unphysical mass configuration";
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Test mode detection functions
////////////////////////////////////////////////////////////////////////////////
TEST_F(FANGTest, ModeDetection)
{
   EXPECT_TRUE(IsPoint(kModePoint));
   EXPECT_FALSE(IsPoint(0.0));
   EXPECT_FALSE(IsPoint(0.5));
   EXPECT_FALSE(IsPoint(-1.0));

   EXPECT_TRUE(IsCircle(kModeCircle));
   EXPECT_FALSE(IsCircle(kModePoint));
   EXPECT_FALSE(IsCircle(0.5));
   EXPECT_FALSE(IsCircle(-1.0));

   EXPECT_TRUE(IsStrip(0.5));
   EXPECT_TRUE(IsStrip(0.1));
   EXPECT_TRUE(IsStrip(1.0));
   EXPECT_FALSE(IsStrip(0.0));
   EXPECT_FALSE(IsStrip(kModePoint));
   EXPECT_FALSE(IsStrip(-0.5));

   EXPECT_TRUE(IsRing(-0.5));
   EXPECT_TRUE(IsRing(-1.0));
   EXPECT_FALSE(IsRing(0.0));
   EXPECT_FALSE(IsRing(0.5));
   EXPECT_FALSE(IsRing(kModePoint));
}

