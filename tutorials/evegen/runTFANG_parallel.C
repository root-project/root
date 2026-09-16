// @(#)root/fang:$Id$
// Author: Arik Kreisel
// Parallelized version using std::thread (no OpenMP required)

////////////////////////////////////////////////////////////////////////////////
/// \file runTFANG_parallel.C
/// \ingroup Physics
/// \brief Parallelized demonstration and validation of FANG using the TFANG class
/// \author Arik Kreisel
///
/// This is the multi-threaded version of runTFANG.C using std::thread.
/// All event generation loops are parallelized for improved performance.
///
/// TFANG is a Monte Carlo tool for efficient event generation in restricted
/// (or full) Lorentz-Invariant Phase Space (LIPS).
///
/// Reference: Horin, I., Kreisel, A. & Alon, O. Focused angular N -body event generator (FANG).
/// J. High Energ. Phys. 2025, 137 (2025). 
/// https://doi.org/10.1007/JHEP12(2025)137 
/// https://arxiv.org/abs/2509.11105 
///
/// Features:
/// - Comparison of TFANG constrained vs TFANG unconstrained with cuts
////////////////////////////////////////////////////////////////////////////////

#include "TFANG.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TF1.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TRandom3.h"
#include "TROOT.h"

// Threading includes
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <condition_variable>
#include <queue>

////////////////////////////////////////////////////////////////////////////////
// Thread-safe Work Queue
////////////////////////////////////////////////////////////////////////////////

class WorkQueue {
public:
   void Push(Int_t item) {
      std::lock_guard<std::mutex> lock(fMutex);
      fQueue.push(item);
      fCV.notify_one();
   }

   bool Pop(Int_t& item) {
      std::lock_guard<std::mutex> lock(fMutex);
      if (fQueue.empty()) return false;
      item = fQueue.front();
      fQueue.pop();
      return true;
   }

   bool Empty() {
      std::lock_guard<std::mutex> lock(fMutex);
      return fQueue.empty();
   }

private:
   std::queue<Int_t> fQueue;
   std::mutex fMutex;
   std::condition_variable fCV;
};

////////////////////////////////////////////////////////////////////////////////
// Structure to hold accumulated results from workers
////////////////////////////////////////////////////////////////////////////////

struct AccumulatorResult {
   Double_t fSumW;
   Double_t fSumW2;
   Int_t fNEvents;

   AccumulatorResult() : fSumW(0.0), fSumW2(0.0), fNEvents(0) {}
};

////////////////////////////////////////////////////////////////////////////////
// Structure to hold TFANG unconstrained with cuts results
////////////////////////////////////////////////////////////////////////////////

struct TFangCutsResult {
   Int_t fNTotalGenerated;
   Int_t fNPassedCuts;

   TFangCutsResult() : fNTotalGenerated(0), fNPassedCuts(0) {}
};

////////////////////////////////////////////////////////////////////////////////
// Structure to hold point calculation results
////////////////////////////////////////////////////////////////////////////////

struct PointResult {
   Double_t fCosTheta;
   Double_t fSigma;
   Double_t fSigmaErr;
};

////////////////////////////////////////////////////////////////////////////////
// Rosenbluth Cross Section for Elastic ep Scattering
////////////////////////////////////////////////////////////////////////////////

Double_t fElastic(Double_t *x, Double_t *par)
{
   using namespace FANG;

   Double_t sigma = 0.0;
   Double_t alpha = 1.0 / 137.0;

   ROOT::Math::XYZVector vDir(TMath::Sqrt(1.0 - x[0] * x[0]), 0.0, x[0]);

   Double_t massProton   = 0.938272029;
   Double_t massElectron = 0.000511;

   ROOT::Math::PxPyPzMVector pProton(0.0, 0.0, 0.0, massProton);
   Double_t kineticE = par[0];
   Double_t gamma = kineticE / massElectron + 1.0;
   Double_t beta = TMath::Sqrt(1.0 - 1.0 / (gamma * gamma));
   ROOT::Math::PxPyPzMVector pElectron(0.0, 0.0, gamma * beta * massElectron, massElectron);

   ROOT::Math::PxPyPzMVector pElectronOut, pMomentumTransfer;
   ROOT::Math::PxPyPzMVector pTotal = pProton + pElectron;

   Double_t mottXS, tau, formGE, formGM, qSquared;

   LongDouble_t massCM = pTotal.M();
   LongDouble_t energyCM = pTotal.E();
   LongDouble_t momCM = pTotal.P();
   LongDouble_t energyCM3 = (massCM * massCM - massProton * massProton +
                             massElectron * massElectron) / (2.0 * massCM);

   LongDouble_t aa = momCM * momCM * x[0] * x[0] - energyCM * energyCM;
   LongDouble_t bb = 2.0 * momCM * x[0] * energyCM3 * massCM;
   LongDouble_t cc = energyCM3 * massCM * energyCM3 * massCM -
                     massElectron * massElectron * energyCM * energyCM;

   if (bb * bb - 4.0 * aa * cc < 0.0) {
      return 0.0;
   }

   LongDouble_t momLAB = (-bb + TMath::Sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa);
   if (momLAB > 0.0) {
      pElectronOut.SetCoordinates(momLAB * vDir.X(), momLAB * vDir.Y(),
                                  momLAB * vDir.Z(), massElectron);
      pMomentumTransfer = pElectronOut - pElectron;
      qSquared = -pMomentumTransfer.M2();
      formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) * (1.0 + qSquared / kDipoleMassSq));
      formGM = kProtonMagneticMoment * formGE;
      tau = qSquared / (4.0 * massProton * massProton);
      mottXS = alpha * alpha / (pElectron.E() * pElectron.E() * (1.0 - x[0]) * (1.0 - x[0])) *
               pElectronOut.E() / pElectron.E() * (1.0 + x[0]) / 2.0;
      sigma = mottXS * ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) +
                        2.0 * tau * formGM * formGM * (1.0 - x[0]) / (1.0 + x[0]));
   }

   momLAB = (-bb - TMath::Sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa);
   if (momLAB > 0.0) {
      pElectronOut.SetCoordinates(momLAB * vDir.X(), momLAB * vDir.Y(),
                                  momLAB * vDir.Z(), massElectron);
      pMomentumTransfer = pElectronOut - pElectron;
      qSquared = -pMomentumTransfer.M2();
      formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) * (1.0 + qSquared / kDipoleMassSq));
      formGM = kProtonMagneticMoment * formGE;
      tau = qSquared / (4.0 * massProton * massProton);
      mottXS = alpha * alpha / (pElectron.E() * pElectron.E() * (1.0 - x[0]) * (1.0 - x[0])) *
               pElectronOut.E() / pElectron.E() * (1.0 + x[0]) / 2.0;
      sigma += mottXS * ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) +
                         2.0 * tau * formGM * formGM * (1.0 - x[0]) / (1.0 + x[0]));
   }

   return sigma;
}

////////////////////////////////////////////////////////////////////////////////
// Worker Functions for Parallel Loops
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
// Worker for Test 1: Full Phase Space Calculation
//------------------------------------------------------------------------------
void WorkerTest1(
   Int_t threadId,
   WorkQueue& workQueue,
   const Int_t kNBody,
   const ROOT::Math::PxPyPzMVector& pTotal,
   const Double_t* masses,
   std::vector<AccumulatorResult>& results,
   std::mutex& resultsMutex
) {
   using namespace FANG;

   AccumulatorResult localResult;

   // Thread-local RNG with unique seed
   TRandom3 rng(threadId + 100);
   
   // Thread-local TFANG instance with thread-local RNG
   TFANG gen(&rng);
   gen.SetDecay(pTotal, kNBody, masses);

   Double_t weight;
   Int_t workItem;

   while (workQueue.Pop(workItem)) {
      if (gen.Generate() == 0) continue;

      for (Int_t i = 0; i < gen.GetNSolutions(); i++) {
         weight = gen.GetWeight(i);
         localResult.fNEvents++;
         localResult.fSumW += weight;
         localResult.fSumW2 += weight * weight;
      }
   }

   // Store results
   {
      std::lock_guard<std::mutex> lock(resultsMutex);
      results.push_back(localResult);
   }
}

//------------------------------------------------------------------------------
// Worker for Test 2: TFANG with Detector Constraints
//------------------------------------------------------------------------------
void WorkerTest2TFANG(
   Int_t threadId,
   WorkQueue& workQueue,
   const Int_t kNBody,
   const Int_t kNDet,
   const ROOT::Math::PxPyPzMVector& pTotal,
   const Double_t* masses,
   const Double_t* omega,
   const Double_t* shape,
   const std::vector<ROOT::Math::XYZVector>& v3DetConst,
   Double_t totalOmega,
   std::vector<TH1D*>& histsE,
   std::vector<TH1D*>& histsCos,
   std::vector<TH1D*>& histsPhi,
   std::vector<AccumulatorResult>& results,
   std::mutex& resultsMutex
) {
   using namespace FANG;

   AccumulatorResult localResult;

   // Thread-local RNG with unique seed
   TRandom3 rng(threadId + 200);
   
   // Thread-local TFANG instance
   TFANG gen(&rng);
   gen.SetDecay(pTotal, kNBody, masses);
   for (Int_t i = 0; i < kNDet; i++) {
      gen.AddConstraint(v3DetConst[i], omega[i], shape[i]);
   }

   Double_t weight;
   Int_t workItem;

   while (workQueue.Pop(workItem)) {
      if (gen.Generate() == 0) continue;

      for (Int_t i = 0; i < gen.GetNSolutions(); i++) {
         weight = gen.GetWeight(i);
         localResult.fNEvents++;
         localResult.fSumW += weight;
         localResult.fSumW2 += weight * weight;

         for (Int_t j = 0; j < kNBody; j++) {
            ROOT::Math::PxPyPzMVector p = gen.GetDecay(i, j);
            histsE[threadId * kNBody + j]->Fill(p.E() - masses[j], weight * totalOmega);
            histsCos[threadId * kNBody + j]->Fill(TMath::Cos(p.Theta()), weight * totalOmega);
            histsPhi[threadId * kNBody + j]->Fill(p.Phi(), weight * totalOmega);
         }
      }
   }

   {
      std::lock_guard<std::mutex> lock(resultsMutex);
      results.push_back(localResult);
   }
}

//------------------------------------------------------------------------------
// Worker for Test 2: TFANG Unconstrained with Cuts
//------------------------------------------------------------------------------
void WorkerTest2TFANGCuts(
   Int_t threadId,
   WorkQueue& workQueue,
   const Int_t kNBody,
   const Int_t kNDet,
   const ROOT::Math::PxPyPzMVector& pTotal,
   const Double_t* masses,
   const Double_t* omega,
   const Double_t* shape,
   const std::vector<TVector3>& tv3,
   Double_t scaleFactor,
   std::vector<TH1D*>& histsE,
   std::vector<TH1D*>& histsCos,
   std::vector<TH1D*>& histsPhi,
   std::vector<TFangCutsResult>& results,
   std::mutex& resultsMutex
) {
   using namespace FANG;

   TFangCutsResult localResult;

   // Thread-local RNG with unique seed
   TRandom3 rng(threadId + 300);
   
   // Thread-local unconstrained TFANG instance
   TFANG gen(&rng);
   gen.SetDecay(pTotal, kNBody, masses);
   // No constraints added for unconstrained generation

   Double_t weight;
   Int_t outsideCut;
   Int_t workItem;

   while (workQueue.Pop(workItem)) {
      if (gen.Generate() == 0) continue;

      localResult.fNTotalGenerated++;

      for (Int_t i = 0; i < gen.GetNSolutions(); i++) {
         weight = gen.GetWeight(i);
         outsideCut = 0;

         // Apply geometric cuts
         for (Int_t j = 0; j < kNDet; j++) {
            ROOT::Math::PxPyPzMVector p = gen.GetDecay(i, j);
            TVector3 pVec(p.Px(), p.Py(), p.Pz());
            
            if (shape[j] == 0.0 &&
                (1.0 - TMath::Cos(tv3[j].Angle(pVec))) > omega[j] / kTwoPi) {
               outsideCut = 1;
            }
            if (shape[j] > 0.0 &&
                (TMath::Abs(tv3[j].Phi() - p.Phi()) > kPi * shape[j] ||
                 TMath::Abs(TMath::Cos(tv3[j].Theta()) - TMath::Cos(p.Theta())) >
                 omega[j] / (4.0 * kPi * shape[j]))) {
               outsideCut = 1;
            }
         }

         if (outsideCut == 1) continue;

         localResult.fNPassedCuts++;

         for (Int_t j = 0; j < kNBody; j++) {
            ROOT::Math::PxPyPzMVector p = gen.GetDecay(i, j);
            histsE[threadId * kNBody + j]->Fill(p.E() - masses[j], weight / scaleFactor);
            histsCos[threadId * kNBody + j]->Fill(TMath::Cos(p.Theta()), weight / scaleFactor);
            histsPhi[threadId * kNBody + j]->Fill(p.Phi(), weight / scaleFactor);
         }
      }
   }

   {
      std::lock_guard<std::mutex> lock(resultsMutex);
      results.push_back(localResult);
   }
}

//------------------------------------------------------------------------------
// Worker for Test 3: Point generation at specific angles
//------------------------------------------------------------------------------
void WorkerTest3Point(
   Int_t threadId,
   WorkQueue& workQueue,
   Int_t nLoop,
   Double_t massElectron,
   Double_t massProton,
   const ROOT::Math::PxPyPzMVector& pTotal2,
   const ROOT::Math::PxPyPzMVector& pTarget,
   const ROOT::Math::PxPyPzMVector& pElectronIn,
   Double_t flux,
   std::vector<PointResult>& pointResults,
   std::mutex& resultsMutex
) {
   using namespace FANG;

   const Int_t kNBody2 = 2;
   Double_t masses2[kNBody2] = {massElectron, massProton};
   Double_t alphaQED = 1.0 / 137.0;

   // Thread-local RNG
   TRandom3 rng(threadId + 500);

   ROOT::Math::XYZVector v3;
   ROOT::Math::PxPyPzMVector pElectronOut, pProtonOut, pMomTransfer;
   Double_t qSquared, formGE, formGM, tau, lambda, ampSquared, weight;

   Int_t angleIdx;
   while (workQueue.Pop(angleIdx)) {
      Double_t sumW = 0.0;
      Double_t sumW2 = 0.0;
      Int_t nEvents = 0;

      Double_t cosTheta = -0.99 + angleIdx * 0.2;
      if (angleIdx == 10) cosTheta = 0.95;

      v3.SetXYZ(TMath::Sqrt(1.0 - cosTheta * cosTheta), 0.0, cosTheta);

      // Thread-local TFANG instance with point constraint
      TFANG gen(&rng);
      gen.SetDecay(pTotal2, kNBody2, masses2);
      gen.AddConstraint(v3, 1.0, kModePoint);

      for (Int_t k = 0; k < nLoop; k++) {
         if (gen.Generate() == 0) continue;

         for (Int_t i = 0; i < gen.GetNSolutions(); i++) {
            weight = gen.GetWeight(i);
            pElectronOut = gen.GetDecay(i, 0);
            pProtonOut = gen.GetDecay(i, 1);
            pMomTransfer = pElectronIn - pElectronOut;
            ROOT::Math::PxPyPzMVector pU = pTarget - pElectronOut;
            qSquared = -pMomTransfer.M2();
            formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) *
                            (1.0 + qSquared / kDipoleMassSq));
            formGM = kProtonMagneticMoment * formGE;
            tau = qSquared / (4.0 * massProton * massProton);
            lambda = (pTotal2.M2() - pU.M2()) / (4.0 * massProton * massProton);

            ampSquared = 16.0 * kPi * kPi * alphaQED * alphaQED / (tau * tau) *
                         ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) *
                          (lambda * lambda - tau * tau - tau) +
                          2.0 * tau * tau * formGM * formGM);

            weight *= ampSquared;
            nEvents++;
            sumW += weight;
            sumW2 += weight * weight;
         }
      }

      PointResult result;
      result.fCosTheta = cosTheta;
      result.fSigma = (nEvents > 0) ? flux * sumW / nEvents : 0.0;
      result.fSigmaErr = (nEvents > 0) ? flux * TMath::Sqrt(sumW2) / nEvents : 0.0;

      {
         std::lock_guard<std::mutex> lock(resultsMutex);
         pointResults.push_back(result);
      }
   }
}

//------------------------------------------------------------------------------
// Worker for Test 3: Full angular distribution
//------------------------------------------------------------------------------
void WorkerTest3Angular(
   Int_t threadId,
   WorkQueue& workQueue,
   Int_t nEventsPerRange,
   Double_t massElectron,
   Double_t massProton,
   const ROOT::Math::PxPyPzMVector& pTotal2,
   const ROOT::Math::PxPyPzMVector& pTarget,
   const ROOT::Math::PxPyPzMVector& pElectronIn,
   Double_t c1,
   Double_t c2,
   TH1D* hXsec,
   TH1D* hNorm,
   TH1D* hCount
) {
   using namespace FANG;

   // Thread-local random generator for direction sampling
   TRandom3 rng(threadId + 600);

   const Int_t kNBody2 = 2;
   Double_t masses2[kNBody2] = {massElectron, massProton};
   Double_t alphaQED = 1.0 / 137.0;

   // Thread-local TFANG instance
   TFANG gen(&rng);
   gen.SetDecay(pTotal2, kNBody2, masses2);

   ROOT::Math::XYZVector v3;
   ROOT::Math::PxPyPzMVector pElectronOut, pProtonOut, pMomTransfer;
   Double_t qSquared, formGE, formGM, tau, lambda, ampSquared, weight;
   Double_t r1, cosTheta, sinTheta, phi;

   Int_t workItem;
   while (workQueue.Pop(workItem)) {
      // Generate r1 with 1/r^2 distribution
      r1 = c1 * c1 * c2 / (c2 * c1 - rng.Uniform(0, 1) * c1 * (c2 - c1));
      cosTheta = 1.0 - r1;
      sinTheta = TMath::Sqrt(1.0 - cosTheta * cosTheta);
      phi = rng.Uniform(0, kTwoPi);

      v3.SetXYZ(sinTheta * TMath::Cos(phi), sinTheta * TMath::Sin(phi), cosTheta);

      // Update constraint for new direction
      gen.ClearConstraints();
      gen.AddConstraint(v3, 1.0, kModePoint);

      if (gen.Generate() == 0) continue;

      for (Int_t i = 0; i < gen.GetNSolutions(); i++) {
         weight = gen.GetWeight(i);
         pElectronOut = gen.GetDecay(i, 0);
         pProtonOut = gen.GetDecay(i, 1);
         pMomTransfer = pElectronIn - pElectronOut;
         ROOT::Math::PxPyPzMVector pU = pTarget - pElectronOut;
         qSquared = -pMomTransfer.M2();
         formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) *
                         (1.0 + qSquared / kDipoleMassSq));
         formGM = kProtonMagneticMoment * formGE;
         tau = qSquared / (4.0 * massProton * massProton);
         lambda = (pTotal2.M2() - pU.M2()) / (4.0 * massProton * massProton);

         ampSquared = 16.0 * kPi * kPi * alphaQED * alphaQED / (tau * tau) *
                      ((formGE * formGE + tau * formGM * formGM) / (1.0 + tau) *
                       (lambda * lambda - tau * tau - tau) +
                       2.0 * tau * tau * formGM * formGM);

         weight *= ampSquared;

         Double_t reweight = r1 * r1 * (c2 - c1) / c1 / c2;
         hXsec->Fill(TMath::Cos(pElectronOut.Theta()), reweight * weight);
         hNorm->Fill(TMath::Cos(pElectronOut.Theta()), reweight);
         hCount->Fill(TMath::Cos(pElectronOut.Theta()), 1.0);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// Main Demonstration Function - Parallelized
////////////////////////////////////////////////////////////////////////////////

void runTFANG()
{
   using namespace FANG;

   // Enable ROOT thread safety - CRITICAL
   ROOT::EnableThreadSafety();

   gStyle->SetOptStat(0);

   Int_t nThreads = std::thread::hardware_concurrency();
   std::cout << "Using std::thread with " << nThreads << " threads" << std::endl;

   Int_t nEvents = 0;

   //==========================================================================
   // Setup for 5-body decay test
   //==========================================================================
   const Int_t kNBody = 5;
   Double_t masses[kNBody] = {1.0, 1.0, 1.0, 1.0, 1.0};
   ROOT::Math::PxPyPzMVector pTotal(0, 0, 5, 12);

   Double_t weight = 0.0;
   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;

   //==========================================================================
   // Test 1: TFANG Full Phase Space Calculation (Parallelized)
   //==========================================================================
   std::cout << "========================================" << std::endl;
   std::cout << "Test 1: Full Phase Space Calculation (TFANG Parallel)" << std::endl;
   std::cout << "========================================" << std::endl;

   Int_t nLoop = 1000000;
   {
      WorkQueue workQueue;
      for (Int_t k = 0; k < nLoop; k++) {
         workQueue.Push(k);
      }

      std::vector<AccumulatorResult> results;
      std::mutex resultsMutex;
      std::vector<std::thread> threads;

      for (Int_t t = 0; t < nThreads; t++) {
         threads.emplace_back(WorkerTest1,
            t, std::ref(workQueue), kNBody, std::cref(pTotal), masses,
            std::ref(results), std::ref(resultsMutex));
      }

      for (auto& t : threads) {
         t.join();
      }

      // Aggregate results
      sumW = 0.0;
      sumW2 = 0.0;
      nEvents = 0;
      for (const auto& r : results) {
         sumW += r.fSumW;
         sumW2 += r.fSumW2;
         nEvents += r.fNEvents;
      }
   }
   Double_t mean = sumW / nEvents;
   Double_t variance = sumW2 / nEvents - mean * mean;

   // Also get phase space using GetPhaseSpace for comparison
   Double_t phaseSpace, phaseSpaceErr;
   {
      TFANG genFull;
      genFull.SetDecay(pTotal, kNBody, masses);
      genFull.GetPhaseSpace(static_cast<Long64_t>(nLoop), phaseSpace, phaseSpaceErr);
   }

   std::cout << "nEvents = " << nEvents << std::endl;
   std::cout << "Total Phase Space from parallel loop = " << sumW / nLoop
             << " +/- " << TMath::Sqrt(variance / nEvents) << std::endl;
   std::cout << "Total Phase Space from GetPhaseSpace = " << phaseSpace
             << " +/- " << phaseSpaceErr << std::endl;

   //==========================================================================
   // Test 2: Partial Phase Space with Detector Constraints (Parallelized)
   //==========================================================================
   std::cout << "\n========================================" << std::endl;
   std::cout << "Test 2: Partial Phase Space (TFANG Parallel)" << std::endl;
   std::cout << "  - TFANG constrained vs TFANG unconstrained with cuts" << std::endl;
   std::cout << "========================================" << std::endl;

   const Int_t kNDet = 3;
   Double_t omega[kNDet];
   Double_t shape[kNDet];

   Double_t detPosX[kNDet - 1] = {0.0, 0.5};
   Double_t detPosY[kNDet - 1] = {0.0, 0.0};
   Double_t detPosZ[kNDet - 1] = {0.5, 0.0};
   Double_t detRadius[kNDet - 1] = {0.2, 0.3};

   std::vector<ROOT::Math::XYZVector> v3Det;
   ROOT::Math::XYZVector v3;
   Double_t radius;
   Double_t totalOmega = 1.0;

   for (Int_t i = 0; i < kNDet - 1; i++) {
      v3.SetXYZ(detPosX[i], detPosY[i], detPosZ[i]);
      v3Det.push_back(v3);
      radius = TMath::Sqrt(v3.Mag2() + detRadius[i] * detRadius[i]);
      omega[i] = kTwoPi * (1-v3.R()/radius);  
      shape[i] = 0.0;
      totalOmega *= omega[i];
   }

   v3.SetXYZ(0, 0.5, 0);
   v3Det.push_back(v3);
   omega[2] = 1.2 * kPi;
   shape[2] = 0.4;
   totalOmega *= omega[2];

   std::cout << "Detector configurations:" << std::endl;
   std::cout << "  Det 1: Circle, Omega = " << omega[0] << " sr" << std::endl;
   std::cout << "  Det 2: Circle, Omega = " << omega[1] << " sr" << std::endl;
   std::cout << "  Det 3: Strip,  Omega = " << omega[2] << " sr" << std::endl;
   std::cout << "  Total solid angle factor = " << totalOmega << std::endl;

   Double_t totalMass = 0.0;
   for (Int_t l = 0; l < kNBody; l++) {
      totalMass += masses[l];
   }

   // Create per-thread histograms for TFANG constrained
   std::vector<TH1D*> hFangE_vec(nThreads * kNBody);
   std::vector<TH1D*> hFangCos_vec(nThreads * kNBody);
   std::vector<TH1D*> hFangPhi_vec(nThreads * kNBody);

   for (Int_t t = 0; t < nThreads; t++) {
      for (Int_t i = 0; i < kNBody; i++) {
         hFangE_vec[t * kNBody + i] = new TH1D(Form("hFangE_%d_%d", t, i), "",
                                               100, 0, pTotal.E() - totalMass);
         hFangCos_vec[t * kNBody + i] = new TH1D(Form("hFangCos_%d_%d", t, i), "", 50, -1, 1);
         hFangPhi_vec[t * kNBody + i] = new TH1D(Form("hFangPhi_%d_%d", t, i), "", 50, -kPi, kPi);
         hFangE_vec[t * kNBody + i]->SetDirectory(0);
         hFangCos_vec[t * kNBody + i]->SetDirectory(0);
         hFangPhi_vec[t * kNBody + i]->SetDirectory(0);
      }
   }

   // Run TFANG constrained (parallel)
   nLoop = 1000000;
   {
      WorkQueue workQueue;
      for (Int_t k = 0; k < nLoop; k++) {
         workQueue.Push(k);
      }

      std::vector<AccumulatorResult> results;
      std::mutex resultsMutex;
      std::vector<std::thread> threads;

      for (Int_t t = 0; t < nThreads; t++) {
         threads.emplace_back(WorkerTest2TFANG,
            t, std::ref(workQueue), kNBody, kNDet,
            std::cref(pTotal), masses, omega, shape, std::cref(v3Det),
            totalOmega,
            std::ref(hFangE_vec), std::ref(hFangCos_vec), std::ref(hFangPhi_vec),
            std::ref(results), std::ref(resultsMutex));
      }

      for (auto& t : threads) {
         t.join();
      }

      // Aggregate results
      sumW = 0.0;
      sumW2 = 0.0;
      nEvents = 0;
      for (const auto& r : results) {
         sumW += r.fSumW;
         sumW2 += r.fSumW2;
         nEvents += r.fNEvents;
      }
   }

   // Merge per-thread histograms for TFANG constrained
   TH1D *hFangE[kNBody];
   TH1D *hFangCos[kNBody];
   TH1D *hFangPhi[kNBody];

   for (Int_t i = 0; i < kNBody; i++) {
      hFangE[i] = new TH1D(Form("hFangE_%d", i), "", 100, 0, pTotal.E() - totalMass);
      hFangE[i]->SetMarkerStyle(20);
      hFangE[i]->SetLineColor(6);
      hFangE[i]->SetMinimum(0);
      hFangE[i]->GetXaxis()->SetTitle(Form("p_{%d} Energy", i + 1));
      hFangE[i]->GetXaxis()->SetTitleSize(0.07);
      hFangE[i]->GetXaxis()->SetLabelSize(0.06);
      hFangE[i]->GetYaxis()->SetLabelSize(0.05);
      hFangE[i]->GetYaxis()->SetTitle("Events");
      hFangE[i]->GetYaxis()->SetTitleSize(0.07);
      hFangE[i]->GetYaxis()->SetTitleOffset(0.5);
      hFangE[i]->GetXaxis()->SetTitleOffset(0.9);

      hFangCos[i] = new TH1D(Form("hFangCos_%d", i), "", 50, -1, 1);
      hFangCos[i]->SetMarkerStyle(20);
      hFangCos[i]->SetLineColor(6);
      hFangCos[i]->SetMinimum(0);
      hFangCos[i]->GetXaxis()->SetTitle(Form("p_{%d} cos(#theta)", i + 1));
      hFangCos[i]->GetXaxis()->SetTitleSize(0.07);
      hFangCos[i]->SetTitleOffset(0.7);
      hFangCos[i]->GetYaxis()->SetTitle("Events");
      hFangCos[i]->GetYaxis()->SetTitleSize(0.07);
      hFangCos[i]->GetYaxis()->SetTitleOffset(0.5);
      hFangCos[i]->GetXaxis()->SetLabelSize(0.06);
      hFangCos[i]->GetYaxis()->SetLabelSize(0.05);
      hFangCos[i]->GetXaxis()->SetTitleOffset(0.9);

      hFangPhi[i] = new TH1D(Form("hFangPhi_%d", i), "", 50, -kPi, kPi);
      hFangPhi[i]->SetMarkerStyle(20);
      hFangPhi[i]->SetLineColor(6);
      hFangPhi[i]->SetMinimum(0);
      hFangPhi[i]->GetXaxis()->SetTitle(Form("p_{%d} #varphi", i + 1));
      hFangPhi[i]->GetXaxis()->SetTitleSize(0.07);
      hFangPhi[i]->SetTitleOffset(0.7);
      hFangPhi[i]->GetYaxis()->SetTitle("Events");
      hFangPhi[i]->GetYaxis()->SetTitleSize(0.07);
      hFangPhi[i]->GetYaxis()->SetTitleOffset(0.5);
      hFangPhi[i]->GetXaxis()->SetLabelSize(0.06);
      hFangPhi[i]->GetYaxis()->SetLabelSize(0.05);
      hFangPhi[i]->GetXaxis()->SetTitleOffset(0.9);

      for (Int_t t = 0; t < nThreads; t++) {
         hFangE[i]->Add(hFangE_vec[t * kNBody + i]);
         hFangCos[i]->Add(hFangCos_vec[t * kNBody + i]);
         hFangPhi[i]->Add(hFangPhi_vec[t * kNBody + i]);
         delete hFangE_vec[t * kNBody + i];
         delete hFangCos_vec[t * kNBody + i];
         delete hFangPhi_vec[t * kNBody + i];
      }
   }

   // Get partial phase space using GetPartialPhaseSpace for comparison
   {
      TFANG genConstrained;
      genConstrained.SetDecay(pTotal, kNBody, masses);
      for (Int_t i = 0; i < kNDet; i++) {
         genConstrained.AddConstraint(v3Det[i], omega[i], shape[i]);
      }
      genConstrained.GetPartialPhaseSpace(static_cast<Long64_t>(nLoop), phaseSpace, phaseSpaceErr);
   }
   
   mean = sumW / nEvents;
   variance = sumW2 / nEvents - mean * mean;
   std::cout << "\nTFANG Constrained Results:" << std::endl;
   std::cout << "  nEvents = " << nEvents << std::endl;
   std::cout << "  Partial Phase Space from parallel loop = " << totalOmega * sumW / nLoop
             << " +/- " << totalOmega * TMath::Sqrt(variance / nEvents) << std::endl;
   std::cout << "  Partial Phase Space from GetPartialPhaseSpace = " << phaseSpace
             << " +/- " << phaseSpaceErr << std::endl;
   std::cout << "  hFangE[0]->Integral() = " << hFangE[0]->Integral() << std::endl;

   // Draw TFANG results
   TCanvas *c1 = new TCanvas("c1", "c1 En", 10, 10, 1800, 1500);
   c1->Divide(2, static_cast<Int_t>(TMath::Floor(kNBody / 2.0 + 0.6)));
   for (Int_t i = 0; i < kNBody; i++) {
      c1->cd(i + 1);
      gPad->SetBottomMargin(0.15);
      hFangE[i]->Draw("hist");
   }

   TCanvas *c2 = new TCanvas("c2", "c2 cos", 10, 10, 1800, 1500);
   c2->Divide(2, static_cast<Int_t>(TMath::Floor(kNBody / 2.0 + 0.6)));
   for (Int_t i = 0; i < kNBody; i++) {
      c2->cd(i + 1);
      gPad->SetBottomMargin(0.15);
      hFangCos[i]->Draw("hist");
   }

   TCanvas *c3 = new TCanvas("c3", "c3 phi", 10, 10, 1800, 1500);
   c3->Divide(2, static_cast<Int_t>(TMath::Floor(kNBody / 2.0 + 0.6)));
   for (Int_t i = 0; i < kNBody; i++) {
      c3->cd(i + 1);
      gPad->SetBottomMargin(0.15);
      hFangPhi[i]->Draw("hist");
   }

   //==========================================================================
   // TFANG Unconstrained with Cuts Comparison (Parallelized)
   //==========================================================================
   std::cout << "\n--- TFANG Unconstrained with Cuts (Parallel) ---" << std::endl;

   // Direction vectors for cut comparison
   std::vector<TVector3> tv3(kNDet);
   for (Int_t i = 0; i < kNDet; i++) {
      tv3[i].SetXYZ(v3Det[i].X(), v3Det[i].Y(), v3Det[i].Z());
      tv3[i] = tv3[i].Unit();
   }

   Double_t scaleFactor = 100.0;

   // Create per-thread histograms for TFANG unconstrained with cuts
   std::vector<TH1D*> hFangCutsE_vec(nThreads * kNBody);
   std::vector<TH1D*> hFangCutsCos_vec(nThreads * kNBody);
   std::vector<TH1D*> hFangCutsPhi_vec(nThreads * kNBody);

   for (Int_t t = 0; t < nThreads; t++) {
      for (Int_t i = 0; i < kNBody; i++) {
         hFangCutsE_vec[t * kNBody + i] = new TH1D(Form("hFangCutsE_%d_%d", t, i), "",
                                                    100, 0, pTotal.E() - totalMass);
         hFangCutsCos_vec[t * kNBody + i] = new TH1D(Form("hFangCutsCos_%d_%d", t, i), "", 50, -1, 1);
         hFangCutsPhi_vec[t * kNBody + i] = new TH1D(Form("hFangCutsPhi_%d_%d", t, i), "", 50, -kPi, kPi);
         hFangCutsE_vec[t * kNBody + i]->SetDirectory(0);
         hFangCutsCos_vec[t * kNBody + i]->SetDirectory(0);
         hFangCutsPhi_vec[t * kNBody + i]->SetDirectory(0);
      }
   }

   // Run TFANG unconstrained with cuts (parallel)
   {
      WorkQueue workQueue;
      for (Int_t k = 0; k < static_cast<Int_t>(nLoop * scaleFactor); k++) {
         workQueue.Push(k);
      }

      std::vector<TFangCutsResult> results;
      std::mutex resultsMutex;
      std::vector<std::thread> threads;

      for (Int_t t = 0; t < nThreads; t++) {
         threads.emplace_back(WorkerTest2TFANGCuts,
            t, std::ref(workQueue), kNBody, kNDet,
            std::cref(pTotal), masses, omega, shape, std::cref(tv3),
            scaleFactor,
            std::ref(hFangCutsE_vec), std::ref(hFangCutsCos_vec), std::ref(hFangCutsPhi_vec),
            std::ref(results), std::ref(resultsMutex));
      }

      for (auto& t : threads) {
         t.join();
      }

      // Aggregate results
      Int_t nTotalGenerated = 0;
      Int_t nPassedCuts = 0;
      for (const auto& r : results) {
         nTotalGenerated += r.fNTotalGenerated;
         nPassedCuts += r.fNPassedCuts;
      }

      std::cout << "  Total events generated: " << nTotalGenerated << std::endl;
      std::cout << "  Events passing cuts: " << nPassedCuts << std::endl;
      if (nTotalGenerated > 0) {
         std::cout << "  Cut efficiency: " << 100.0 * nPassedCuts / nTotalGenerated << "%" << std::endl;
      }
   }

   // Merge per-thread histograms for TFANG unconstrained with cuts
   TH1D *hFangCutsE[kNBody];
   TH1D *hFangCutsCos[kNBody];
   TH1D *hFangCutsPhi[kNBody];

   for (Int_t i = 0; i < kNBody; i++) {
      hFangCutsE[i] = new TH1D(Form("hFangCutsE_%d", i), "", 100, 0, pTotal.E() - totalMass);
      hFangCutsCos[i] = new TH1D(Form("hFangCutsCos_%d", i), "", 50, -1, 1);
      hFangCutsPhi[i] = new TH1D(Form("hFangCutsPhi_%d", i), "", 50, -kPi, kPi);
      hFangCutsE[i]->SetMarkerStyle(21);
      hFangCutsE[i]->SetMarkerColor(kBlue);
      hFangCutsCos[i]->SetMarkerStyle(21);
      hFangCutsCos[i]->SetMarkerColor(kBlue);
      hFangCutsPhi[i]->SetMarkerStyle(21);
      hFangCutsPhi[i]->SetMarkerColor(kBlue);

      for (Int_t t = 0; t < nThreads; t++) {
         hFangCutsE[i]->Add(hFangCutsE_vec[t * kNBody + i]);
         hFangCutsCos[i]->Add(hFangCutsCos_vec[t * kNBody + i]);
         hFangCutsPhi[i]->Add(hFangCutsPhi_vec[t * kNBody + i]);
         delete hFangCutsE_vec[t * kNBody + i];
         delete hFangCutsCos_vec[t * kNBody + i];
         delete hFangCutsPhi_vec[t * kNBody + i];
      }
   }

   std::cout << "  hFangCutsE[0]->Integral() = " << hFangCutsE[0]->Integral() << std::endl;

   // Setup legends
   TLegend *leg[3 * kNBody];
   for (Int_t i = 0; i < kNBody * 3; i++) {
      leg[i] = new TLegend(0.52, 0.62, 0.85, 0.88);
   }

   // Adjust legend positions for some plots
   leg[10] = new TLegend(0.12, 0.12, 0.45, 0.38);
   leg[11] = new TLegend(0.56, 0.62, 0.89, 0.88);
   leg[12] = new TLegend(0.12, 0.62, 0.45, 0.88);
   for (Int_t i = 5; i <= 9; i++) {
      leg[i] = new TLegend(0.12, 0.62, 0.45, 0.88);
   }

   for (Int_t i = 0; i < kNBody; i++) {
      leg[i]->AddEntry(hFangE[i], "TFANG constrained", "l");
      leg[i]->AddEntry(hFangCutsE[i], "TFANG unconstrained with cuts", "p");

      leg[i + kNBody]->AddEntry(hFangCos[i], "TFANG constrained", "l");
      leg[i + kNBody]->AddEntry(hFangCutsCos[i], "TFANG unconstrained with cuts", "p");

      leg[i + 2 * kNBody]->AddEntry(hFangPhi[i], "TFANG constrained", "l");
      leg[i + 2 * kNBody]->AddEntry(hFangCutsPhi[i], "TFANG unconstrained with cuts", "p");
   }

   // Overlay comparison results
   for (Int_t i = 0; i < kNBody; i++) {
      c1->cd(i + 1);
      hFangCutsE[i]->DrawCopy("ep same");
      leg[i]->Draw();

      c2->cd(i + 1);
      hFangCutsCos[i]->DrawCopy("ep same");
      leg[i + kNBody]->Draw();

      c3->cd(i + 1);
      hFangCutsPhi[i]->DrawCopy("ep same");
      leg[i + 2 * kNBody]->Draw();
   }

   //==========================================================================
   // Test 3: Elastic ep Scattering Cross Section (Parallelized)
   //==========================================================================
   std::cout << "\n========================================" << std::endl;
   std::cout << "Test 3: Elastic ep Differential Cross Section (TFANG Parallel)" << std::endl;
   std::cout << "========================================" << std::endl;

   const Int_t kNBody2 = 2;
   nLoop = 100000;

   Double_t massElectron = 0.000511;
   Double_t massProton = 0.938272029;

   ROOT::Math::PxPyPzMVector pTarget(0.0, 0.0, 0.0, massProton);
   Double_t kineticE = 3.0;
   Double_t gamma = kineticE / massElectron + 1.0;
   Double_t beta = TMath::Sqrt(1.0 - 1.0 / (gamma * gamma));
   ROOT::Math::PxPyPzMVector pBeam(0.0, 0.0, gamma * beta * massElectron, massElectron);
   ROOT::Math::PxPyPzMVector pTotal2 = pBeam + pTarget;

   ROOT::Math::PxPyPzMVector pElectronIn = pBeam;
   Double_t flux = 1.0 / (16.0 * kPi * kPi *
                          TMath::Sqrt(pElectronIn.Dot(pTarget) * pElectronIn.Dot(pTarget) -
                                      massElectron * massElectron * massProton * massProton));

   TF1 *fRosenbluth = new TF1("fRosenbluth", fElastic, -1, 0.9992, 1);
   Double_t parElastic[1] = {kineticE};
   fRosenbluth->SetParameters(parElastic);

   //==========================================================================
   // TFANG Point Generation: Differential Cross Section at Specific Angles
   //==========================================================================
   Double_t sigmaArr[11];
   Double_t sigmaErrArr[11];
   Double_t cosThetaArr[11];
   Double_t cosThetaErrArr[11];

   std::cout << "\nCalculating differential cross section at specific angles:" << std::endl;

   {
      WorkQueue workQueue;
      for (Int_t l = 0; l < 11; l++) {
         workQueue.Push(l);
      }

      std::vector<PointResult> pointResults;
      std::mutex resultsMutex;
      std::vector<std::thread> threads;

      for (Int_t t = 0; t < nThreads; t++) {
         threads.emplace_back(WorkerTest3Point,
            t, std::ref(workQueue), nLoop,
            massElectron, massProton,
            std::cref(pTotal2), std::cref(pTarget), std::cref(pElectronIn),
            flux, std::ref(pointResults), std::ref(resultsMutex));
      }

      for (auto& t : threads) {
         t.join();
      }

      // Sort results by cosTheta and extract
      std::sort(pointResults.begin(), pointResults.end(),
                [](const PointResult& a, const PointResult& b) {
                   return a.fCosTheta < b.fCosTheta;
                });

      for (Int_t l = 0; l < 11; l++) {
         cosThetaArr[l] = pointResults[l].fCosTheta;
         cosThetaErrArr[l] = 0.0;
         sigmaArr[l] = pointResults[l].fSigma;
         sigmaErrArr[l] = pointResults[l].fSigmaErr;

         std::cout << "  cos(theta) = " << cosThetaArr[l]
                   << ": dsigma/dOmega = " << sigmaArr[l] << " +/- " << sigmaErrArr[l]
                   << " (TFANG/Rosenbluth = " << sigmaArr[l] / fRosenbluth->Eval(cosThetaArr[l]) << ")"
                   << std::endl;
      }
   }

   TGraphErrors *grElastic = new TGraphErrors(11, cosThetaArr, sigmaArr, cosThetaErrArr, sigmaErrArr);
   grElastic->SetMarkerStyle(20);
   grElastic->SetMarkerSize(1.3);

   //==========================================================================
   // TFANG Event Generation: Full Angular Distribution (Parallelized)
   //==========================================================================
   std::cout << "\nGenerating full angular distribution..." << std::endl;

   struct Range_t {
      Double_t fC1;
      Double_t fC2;
   };
   Range_t ranges[4] = {{1.0, 2.0}, {0.4, 1.0}, {0.12, 0.4}, {0.01, 0.12}};

   // Create per-thread histograms for each range
   std::vector<TH1D*> hXsec_vec(nThreads);
   std::vector<TH1D*> hNorm_vec(nThreads);
   std::vector<TH1D*> hCount_vec(nThreads);

   for (Int_t t = 0; t < nThreads; t++) {
      hXsec_vec[t] = new TH1D(Form("hXsec_%d", t), "", 440, -1.1, 1.1);
      hNorm_vec[t] = new TH1D(Form("hNorm_%d", t), "", 440, -1.1, 1.1);
      hCount_vec[t] = new TH1D(Form("hCount_%d", t), "", 440, -1.1, 1.1);
      hXsec_vec[t]->SetDirectory(0);
      hNorm_vec[t]->SetDirectory(0);
      hCount_vec[t]->SetDirectory(0);
   }

   for (Int_t rangeIdx = 0; rangeIdx < 4; rangeIdx++) {
      Double_t c1 = ranges[rangeIdx].fC1;
      Double_t c2 = ranges[rangeIdx].fC2;

      std::cout << "  Range " << rangeIdx + 1 << ": cos(theta) in ["
                << 1.0 - c2 << ", " << 1.0 - c1 << "]" << std::endl;

      WorkQueue workQueue;
      Int_t nEventsPerRange = 1000000;
      for (Int_t k = 0; k < nEventsPerRange; k++) {
         workQueue.Push(k);
      }

      std::vector<std::thread> threads;
      for (Int_t t = 0; t < nThreads; t++) {
         threads.emplace_back(WorkerTest3Angular,
            t, std::ref(workQueue), nEventsPerRange,
            massElectron, massProton,
            std::cref(pTotal2), std::cref(pTarget), std::cref(pElectronIn),
            c1, c2,
            hXsec_vec[t], hNorm_vec[t], hCount_vec[t]);
      }

      for (auto& t : threads) {
         t.join();
      }
   }

   // Merge per-thread histograms
   TH1D *hXsec = new TH1D("hXsec", "hXsec", 440, -1.1, 1.1);
   TH1D *hNorm = new TH1D("hNorm", "hNorm", 440, -1.1, 1.1);
   TH1D *hCount = new TH1D("hCount", "hCount", 440, -1.1, 1.1);
   TH1D *hError = new TH1D("hError", "hError", 440, -1.1, 1.1);
   hXsec->SetMinimum(1E-17);
   hNorm->SetMinimum(1E-17);
   hError->SetMinimum(0.999);

   for (Int_t t = 0; t < nThreads; t++) {
      hXsec->Add(hXsec_vec[t]);
      hNorm->Add(hNorm_vec[t]);
      hCount->Add(hCount_vec[t]);
      delete hXsec_vec[t];
      delete hNorm_vec[t];
      delete hCount_vec[t];
   }

   // Scale and compute errors
   Double_t scaleXsec = flux * 2.0 / hXsec->GetBinWidth(2) / hNorm->Integral();

   for (Int_t l = 1; l <= hXsec->GetNbinsX(); l++) {
      Double_t signal = hXsec->GetBinContent(l);
      Double_t error = hXsec->GetBinError(l);
      Double_t entries = hCount->GetBinContent(l);

      if (entries > 0) {
         hError->SetBinContent(l, signal / error / TMath::Sqrt(entries));
      }
      hXsec->SetBinContent(l, signal * scaleXsec);
      hXsec->SetBinError(l, error * scaleXsec);
   }

   hXsec->SetYTitle("#frac{d#sigma}{d#Omega}(ep -> ep)    [GeV^{-2}]");
   hXsec->SetXTitle("cos(#theta_{LAB})");
   hXsec->SetTitle("Electron Energy E=3 GeV");

   TLegend *legFinal = new TLegend(0.12, 0.68, 0.42, 0.88);

   TCanvas *cFinal = new TCanvas("cFinal", "cFinal his", 10, 10, 1800, 1500);
   gPad->SetLogy();
   TH1F *vFrame = gPad->DrawFrame(-1.2, 5E-10, 1, 5E-3);
   hXsec->Draw("hist same");
   grElastic->Draw("P");
   fRosenbluth->Draw("same");

   legFinal->AddEntry(hXsec, "TFANG event generation", "l");
   legFinal->AddEntry(grElastic, "TFANG point calculation", "p");
   legFinal->AddEntry(fRosenbluth, "Rosenbluth cross section", "l");
   legFinal->Draw();

   vFrame->SetYTitle("#frac{d#sigma}{d#Omega}(ep -> ep)    [GeV^{-2}]");
   vFrame->SetXTitle("cos(#theta_{LAB})");
   vFrame->SetTitle("Electron Energy E=3 GeV");

   TCanvas *cDiag = new TCanvas("cDiag", "cDiag Wi error", 10, 10, 1800, 1500);
   cDiag->Divide(2, 1);
   cDiag->cd(1);
   hNorm->Draw("hist");
   cDiag->cd(2);
   hCount->Draw("hist");

   std::cout << "\n========================================" << std::endl;
   std::cout << "runTFANG() completed successfully (parallel version)" << std::endl;
   std::cout << "J. High Energ. Phys. 2025, 137 (2025)" << std::endl;
   std::cout << "https://doi.org/10.1007/JHEP12(2025)137" << std::endl;
   std::cout << "========================================" << std::endl;
}
