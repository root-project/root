// @(#)root/fang:$Id$
// Author: Arik Kreisel

/**
 * \file runFANG.C
 * \brief Focused Angular N-body event Generator (FANG)
 * \authors: Arik Kreisel and Itay Horin 
 *
 * FANG is a Monte Carlo tool for efficient event generation in restricted
 * (or full) Lorentz-Invariant Phase Space (LIPS). Unlike conventional approaches
 * that always sample the full 4pi solid angle, FANG can also directly generates 
 * events in which selected final-state particles are constrained to fixed 
 * directions or finite angular regions in the laboratory frame.
 *
 * Reference: Horin, I., Kreisel, A. & Alon, O. Focused angular N -body event generator (FANG).
 * J. High Energ. Phys. 2025, 137 (2025). 
 * https://doi.org/10.1007/JHEP12(2025)13 
 * https://arxiv.org/abs/2509.11105 
* This file contains:
* 1. Rosenbluth cross section function for elastic ep scattering
* 2. runFANG() - main demonstration function that validates FANG against:
*    - Full phase space calculation
*    - Partial phase space with detector constraints (vs FANG unconstrained with cuts)
*    - Partial phase space with detector constraints (vs TGenPhaseSpace) - optional
*    - Elastic ep differential cross section (vs Rosenbluth formula)

 */

#include "FANG.h"

#include "TStyle.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TF1.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TGenPhaseSpace.h"
#include "TLorentzVector.h"
#include "TVector3.h"

////////////////////////////////////////////////////////////////////////////////
// Configuration: Set to false to skip TGenPhaseSpace comparison
////////////////////////////////////////////////////////////////////////////////
const Bool_t kRunTGenPhaseSpace = true;

////////////////////////////////////////////////////////////////////////////////
// Rosenbluth Cross Section for Elastic ep Scattering
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate Rosenbluth cross section for elastic ep scattering
///
/// This is a ROOT TF1-compatible function that calculates the differential
/// cross section dsigma/dOmega for elastic electron-proton scattering.
///
/// \param[in] x Array where x[0] = cos(theta_lab)
/// \param[in] par Array where par[0] = electron kinetic energy W [GeV]
/// \return Differential cross section in GeV^-2
////////////////////////////////////////////////////////////////////////////////
Double_t fElastic(Double_t *x, Double_t *par)
{
   using namespace FANG;

   Double_t sigma = 0.0;
   Double_t alpha = 1.0 / 137.0;

   // Direction unit vector for scattering angle
   ROOT::Math::XYZVector vDir(TMath::Sqrt(1.0 - x[0] * x[0]), 0.0, x[0]);

   // Particle masses
   Double_t massProton   = 0.938272029;  // proton mass in GeV
   Double_t massElectron = 0.000511;     // electron mass in GeV

   // Setup kinematics
   ROOT::Math::PxPyPzMVector pProton(0.0, 0.0, 0.0, massProton);  // proton at rest
   Double_t kineticE = par[0];                                     // electron kinetic energy
   Double_t gamma = kineticE / massElectron + 1.0;
   Double_t beta = TMath::Sqrt(1.0 - 1.0 / (gamma * gamma));
   ROOT::Math::PxPyPzMVector pElectron(0.0, 0.0, gamma * beta * massElectron, massElectron);

   ROOT::Math::PxPyPzMVector pElectronOut, pMomentumTransfer;
   ROOT::Math::PxPyPzMVector pTotal = pProton + pElectron;  // total 4-momentum

   Double_t mottXS, tau, formGE, formGM, qSquared;

   // CM frame quantities
   LongDouble_t massCM = pTotal.M();
   LongDouble_t energyCM = pTotal.E();
   LongDouble_t momCM = pTotal.P();
   LongDouble_t energyCM3 = (massCM * massCM - massProton * massProton + 
                             massElectron * massElectron) / (2.0 * massCM);

   // Quadratic equation coefficients
   LongDouble_t aa = momCM * momCM * x[0] * x[0] - energyCM * energyCM;
   LongDouble_t bb = 2.0 * momCM * x[0] * energyCM3 * massCM;
   LongDouble_t cc = energyCM3 * massCM * energyCM3 * massCM - 
                     massElectron * massElectron * energyCM * energyCM;

   // Check for physical solutions
   if (bb * bb - 4.0 * aa * cc < 0.0) {
      return 0.0;
   }

   // First solution
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

   // Second solution
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
// Main Demonstration Function
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Main demonstration and validation function for FANG
///
/// Performs three validation tests:
/// 1. Full phase space calculation for 5-body decay
/// 2. Partial phase space with 3 detector constraints, compared to:
///    - FANG unconstrained (nDet=0) with geometric cuts
///    - TGenPhaseSpace with cuts (if kRunTGenPhaseSpace is true)
/// 3. Elastic ep scattering differential cross section vs Rosenbluth formula
////////////////////////////////////////////////////////////////////////////////
void runFANG()
{
   using namespace FANG;

   gStyle->SetOptStat(0);

   // Create random number generator with reproducible seed
   TRandom3 rng(12345);

   Int_t nEvents = 0;

   //==========================================================================
   // Setup for 5-body decay test
   //==========================================================================
   const Int_t kNBody = 5;
   Double_t masses[kNBody] = {1.0, 1.0, 1.0, 1.0, 1.0};  // outgoing masses
   ROOT::Math::PxPyPzMVector pTotal(0, 0, 5, 12);        // total 4-momentum

   std::vector<ROOT::Math::XYZVector> v3Det;
   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;
   std::vector<ROOT::Math::PxPyPzMVector> vecP;

   Double_t weight = 0.0;
   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;
   Int_t eventStatus;

   //==========================================================================
   // Test 1: FANG Full Phase Space Calculation
   //==========================================================================
   std::cout << "========================================" << std::endl;
   std::cout << "Test 1: Full Phase Space Calculation" << std::endl;
   std::cout << "========================================" << std::endl;

   Double_t nLoop = 1E6;
   Double_t omega0[1];    // Empty arrays for no constraints
   Double_t shape0[1];

   for (Int_t k = 0; k < nLoop; k++) {
      vecVecP.clear();
      vecWi.clear();
      eventStatus = GenFANG(kNBody, pTotal, masses, omega0, shape0, v3Det, vecVecP, vecWi, &rng);
      if (!eventStatus) continue;

      for (size_t i = 0; i < vecVecP.size(); i++) {
         vecP = vecVecP[i];
         weight = vecWi[i];
         nEvents++;
         sumW += weight;
         sumW2 += weight * weight;
      }
   }

   std::cout << "nEvents = " << nEvents << std::endl;
   std::cout << "Total Phase Space = " << sumW / nEvents
             << " +/- " << TMath::Sqrt(sumW2) / nEvents << std::endl;

   //==========================================================================
   // Test 2: Partial Phase Space with Detector Constraints
   //==========================================================================
   std::cout << "\n========================================" << std::endl;
   std::cout << "Test 2: Partial Phase Space" << std::endl;
   std::cout << "  - FANG constrained vs FANG unconstrained with cuts" << std::endl;
   if (kRunTGenPhaseSpace) {
      std::cout << "  - FANG constrained vs TGenPhaseSpace with cuts" << std::endl;
   }
   std::cout << "========================================" << std::endl;

   const Int_t kNDet = 3;
   Double_t omega[kNDet];
   Double_t shape[kNDet];

   // Detector positions and radii
   Double_t detPosX[kNDet - 1] = {0.0, 0.5};
   Double_t detPosY[kNDet - 1] = {0.0, 0.0};
   Double_t detPosZ[kNDet - 1] = {0.5, 0.0};
   Double_t detRadius[kNDet - 1] = {0.2, 0.3};

   ROOT::Math::XYZVector v3;
   Double_t radius;
   Double_t totalOmega = 1.0;

   // Setup first two detectors (circular)
   for (Int_t i = 0; i < kNDet - 1; i++) {
      v3.SetXYZ(detPosX[i], detPosY[i], detPosZ[i]);
      v3Det.push_back(v3);
      radius = TMath::Sqrt(v3.Mag2() + detRadius[i] * detRadius[i]);
      omega[i] = kTwoPi * radius * (radius - v3.R());
      shape[i] = 0.0;  // Circle generation
      totalOmega *= omega[i];
   }

   // Setup third detector (strip)
   v3.SetXYZ(0, 0.5, 0);
   v3Det.push_back(v3);
   omega[2] = 1.2 * kPi;
   shape[2] = 0.4;  // Strip generation
   totalOmega *= omega[2];

   std::cout << "Detector configurations:" << std::endl;
   std::cout << "  Det 1: Circle, Omega = " << omega[0] << " sr" << std::endl;
   std::cout << "  Det 2: Circle, Omega = " << omega[1] << " sr" << std::endl;
   std::cout << "  Det 3: Strip,  Omega = " << omega[2] << " sr" << std::endl;
   std::cout << "  Total solid angle factor = " << totalOmega << std::endl;

   // Calculate total available mass for kinetic energy
   Double_t totalMass = 0.0;
   for (Int_t l = 0; l < kNBody; l++) {
      totalMass += masses[l];
   }

   // Create histograms for FANG results
   TH1D *hFangE[kNBody];
   TH1D *hFangCos[kNBody];
   TH1D *hFangPhi[kNBody];
   TH1D *hFullE[kNBody];
   TH1D *hFullCos[kNBody];

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

      hFullE[i] = new TH1D(Form("hFullE_%d", i), "hFullE", 100, 0, pTotal.E() - totalMass);
      hFullCos[i] = new TH1D(Form("hFullCos_%d", i), "hFullCos", 50, -1, 1);
      hFullE[i]->SetMarkerStyle(20);
      hFullCos[i]->SetMarkerStyle(20);
   }

   // Create histograms for FANG unconstrained with cuts comparison
   TH1D *hFangCutsE[kNBody];
   TH1D *hFangCutsCos[kNBody];
   TH1D *hFangCutsPhi[kNBody];

   for (Int_t i = 0; i < kNBody; i++) {
      hFangCutsE[i] = new TH1D(Form("hFangCutsE_%d", i), "hFangCutsE", 100, 0, pTotal.E() - totalMass);
      hFangCutsCos[i] = new TH1D(Form("hFangCutsCos_%d", i), "hFangCutsCos", 50, -1, 1);
      hFangCutsPhi[i] = new TH1D(Form("hFangCutsPhi_%d", i), "hFangCutsPhi", 50, -kPi, kPi);
      hFangCutsE[i]->SetMarkerStyle(21);
      hFangCutsE[i]->SetMarkerColor(kBlue);
      hFangCutsCos[i]->SetMarkerStyle(21);
      hFangCutsCos[i]->SetMarkerColor(kBlue);
      hFangCutsPhi[i]->SetMarkerStyle(21);
      hFangCutsPhi[i]->SetMarkerColor(kBlue);
   }

   // Create histograms for TGenPhaseSpace (CERN/GENBOD) comparison
   TH1D *hGenbodE[kNBody];
   TH1D *hGenbodCos[kNBody];
   TH1D *hGenbodPhi[kNBody];

   for (Int_t i = 0; i < kNBody; i++) {
      hGenbodE[i] = new TH1D(Form("hGenbodE_%d", i), "hGenbodE", 100, 0, pTotal.E() - totalMass);
      hGenbodCos[i] = new TH1D(Form("hGenbodCos_%d", i), "hGenbodCos", 50, -1, 1);
      hGenbodPhi[i] = new TH1D(Form("hGenbodPhi_%d", i), "hGenbodPhi", 50, -kPi, kPi);
      hGenbodE[i]->SetMarkerStyle(20);
      hGenbodCos[i]->SetMarkerStyle(20);
      hGenbodPhi[i]->SetMarkerStyle(20);
   }

   // Run FANG with detector constraints
   weight = 0.0;
   sumW = 0.0;
   sumW2 = 0.0;
   nEvents = 0;
   nLoop = 1E5;

   TH1D *hWeight = new TH1D("hWeight", "hWeight", 100, 0, 10);

   for (Int_t k = 0; k < nLoop; k++) {
      vecVecP.clear();
      vecWi.clear();
/**
 * GenFANG
 * \param[in] nBody Number of outgoing particles
 * \param[in] S Total 4-momentum of the system
 * \param[in] masses Array of outgoing particle masses [GeV], length nBody
 * \param[in] Om Array of solid angles for constrained detectors [sr]
 * \param[in] Ratio Array of shape parameters for each detector:
 *                  - = 2: Point generation (fixed direction)
 *                  - = 0: Circle generation (uniform in cone)
 *                  - 0 < shape[] <= 1: Strip generation (rectangular region)
 *                              Dphi = shape[] * TwoPi;
 *                              Dcos = Omega / Dphi;
 *                  - < 0: Ring generation (fixed theta, uniform phi)
 * \param[in] V3Det Vector of direction vectors for constrained detectors
 * \param[out] VecVecP Output: vector of 4-momenta vectors for each solution
 * \param[out] vecWi Output: weight for each solution
 * \return 1 on success, 0 if no physical solution exists
 */

       eventStatus = GenFANG(kNBody, pTotal, masses, omega, shape, v3Det, vecVecP, vecWi, &rng);
      if (!eventStatus) continue;

      for (size_t i = 0; i < vecVecP.size(); i++) {
         vecP = vecVecP[i];
         weight = vecWi[i];
         nEvents++;
         sumW += weight;
         sumW2 += weight * weight;

         for (size_t j = 0; j < vecP.size(); j++) {
            hFangE[j]->Fill(vecP[j].E() - masses[j], weight * totalOmega);
            hFangCos[j]->Fill(TMath::Cos(vecP[j].Theta()), weight * totalOmega);
            hFangPhi[j]->Fill(vecP[j].Phi(), weight * totalOmega);
         }
      }
   }

   std::cout << "\nFANG Constrained Results:" << std::endl;
   std::cout << "  nEvents = " << nEvents << std::endl;
   std::cout << "  Partial Phase Space = " << totalOmega * sumW / nEvents
             << " +/- " << totalOmega * TMath::Sqrt(sumW2) / nEvents << std::endl;
   std::cout << "  hFangE[0]->Integral() = " << hFangE[0]->Integral() << std::endl;

   // Draw FANG results
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
   // FANG Unconstrained (nDet=0) with Cuts Comparison
   //==========================================================================
   std::cout << "\n--- FANG Unconstrained (nDet=0) with Cuts ---" << std::endl;

   // Direction vectors for cut comparison
   TVector3 tv3[kNDet];
   for (Int_t i = 0; i < kNDet; i++) {
      tv3[i].SetXYZ(v3Det[i].X(), v3Det[i].Y(), v3Det[i].Z());
      tv3[i] = tv3[i].Unit();
   }

   Double_t scaleFactor = 100.0;  // Need more events since most will be rejected by cuts
   Int_t outsideCut = 0;
   Int_t nPassedCuts = 0;
   Int_t nTotalGenerated = 0;

   // Clear detector vectors for unconstrained generation
   std::vector<ROOT::Math::XYZVector> v3DetEmpty;

   for (Int_t k = 0; k < nLoop * scaleFactor; k++) {
      vecVecP.clear();
      vecWi.clear();
      
      // Generate unconstrained events (nDet=0)
      eventStatus = GenFANG(kNBody, pTotal, masses, omega0, shape0, v3DetEmpty, vecVecP, vecWi, &rng);
      if (!eventStatus) continue;

      nTotalGenerated++;

      for (size_t i = 0; i < vecVecP.size(); i++) {
         vecP = vecVecP[i];
         weight = vecWi[i];
         outsideCut = 0;

         // Apply geometric cuts (same as TGenPhaseSpace comparison)
         for (Int_t j = 0; j < kNDet; j++) {
            TVector3 pVec(vecP[j].Px(), vecP[j].Py(), vecP[j].Pz());
            
            if (shape[j] == 0.0 &&
                (1.0 - TMath::Cos(tv3[j].Angle(pVec))) > omega[j] / kTwoPi) {
               outsideCut = 1;
            }
            if (shape[j] > 0.0 &&
                (TMath::Abs(tv3[j].Phi() - vecP[j].Phi()) > kPi * shape[j] ||
                 TMath::Abs(TMath::Cos(tv3[j].Theta()) - TMath::Cos(vecP[j].Theta())) >
                 omega[j] / (4.0 * kPi * shape[j]))) {
               outsideCut = 1;
            }
         }

         if (outsideCut == 1) continue;

         nPassedCuts++;

         for (Int_t j = 0; j < kNBody; j++) {
            hFangCutsE[j]->Fill(vecP[j].E() - masses[j], weight/scaleFactor);
            hFangCutsCos[j]->Fill(TMath::Cos(vecP[j].Theta()), weight/scaleFactor);
            hFangCutsPhi[j]->Fill(vecP[j].Phi(), weight/scaleFactor);
         }
      }
   }

   std::cout << "  Total events generated: " << nTotalGenerated << std::endl;
   std::cout << "  Events passing cuts: " << nPassedCuts << std::endl;
   if (nTotalGenerated > 0) {
      std::cout << "  Cut efficiency: " << 100.0 * nPassedCuts / nTotalGenerated << "%" << std::endl;
   }
   std::cout << "  hFangCutsE[0]->Integral() = " << hFangCutsE[0]->Integral() << std::endl;

   //==========================================================================
   // TGenPhaseSpace comparison (GENBOD with cuts) - Optional
   //==========================================================================
   if (kRunTGenPhaseSpace) {
      std::cout << "\n--- TGenPhaseSpace (GENBOD) with Cuts ---" << std::endl;

      // TGenPhaseSpace uses gRandom internally - set up dedicated RNG
      TRandom3 genbodRng(54321);
      TRandom* savedRandom = gRandom;
      gRandom = &genbodRng;

      TLorentzVector pTotalCern;
      pTotalCern.SetPxPyPzE(0, 0, 5, 13);
      TGenPhaseSpace genPhaseSpace;
      Double_t genWeight;
      genPhaseSpace.SetDecay(pTotalCern, kNBody, masses);

      Double_t normFactor = 2050032.6;  // Normalizing factor

      for (Int_t k = 0; k < nLoop * scaleFactor; k++) {
         genWeight = genPhaseSpace.Generate() / scaleFactor * normFactor;
         outsideCut = 0;

         // Apply geometric cuts
         for (Int_t i = 0; i < kNDet; i++) {
            if (shape[i] == 0.0 &&
                (1.0 - TMath::Cos(tv3[i].Angle(genPhaseSpace.GetDecay(i)->Vect()))) > omega[i] / kTwoPi) {
               outsideCut = 1;
            }
            if (shape[i] > 0.0 &&
                (TMath::Abs(tv3[i].Phi() - genPhaseSpace.GetDecay(i)->Phi()) > kPi * shape[i] ||
                 TMath::Abs(TMath::Cos(tv3[i].Theta()) - TMath::Cos(genPhaseSpace.GetDecay(i)->Theta())) >
                 omega[i] / (4.0 * kPi * shape[i]))) {
               outsideCut = 1;
            }
         }

         if (outsideCut == 1) continue;

         for (Int_t i = 0; i < kNBody; i++) {
            hGenbodE[i]->Fill(genPhaseSpace.GetDecay(i)->E() - masses[i], genWeight);
            hGenbodCos[i]->Fill(TMath::Cos(genPhaseSpace.GetDecay(i)->Theta()), genWeight);
            hGenbodPhi[i]->Fill(genPhaseSpace.GetDecay(i)->Phi(), genWeight);
         }
      }

      // Restore original gRandom
      gRandom = savedRandom;

      std::cout << "  hGenbodE[0]->Integral() = " << hGenbodE[0]->Integral() << std::endl;
   }

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
      leg[i]->AddEntry(hFangE[i], "FANG constrained", "l");
      leg[i]->AddEntry(hFangCutsE[i], "FANG nDet=0 with cuts", "p");
      if (kRunTGenPhaseSpace) {
         leg[i]->AddEntry(hGenbodE[i], "GENBOD with cuts", "p");
      }

      leg[i + kNBody]->AddEntry(hFangCos[i], "FANG constrained", "l");
      leg[i + kNBody]->AddEntry(hFangCutsCos[i], "FANG nDet=0 with cuts", "p");
      if (kRunTGenPhaseSpace) {
         leg[i + kNBody]->AddEntry(hGenbodCos[i], "GENBOD with cuts", "p");
      }

      leg[i + 2 * kNBody]->AddEntry(hFangPhi[i], "FANG constrained", "l");
      leg[i + 2 * kNBody]->AddEntry(hFangCutsPhi[i], "FANG nDet=0 with cuts", "p");
      if (kRunTGenPhaseSpace) {
         leg[i + 2 * kNBody]->AddEntry(hGenbodPhi[i], "GENBOD with cuts", "p");
      }
   }

   // Overlay comparison results
   for (Int_t i = 0; i < kNBody; i++) {
      c1->cd(i + 1);
      hFangCutsE[i]->DrawCopy("ep same");
      if (kRunTGenPhaseSpace) {
         hGenbodE[i]->DrawCopy("ep same");
      }
      leg[i]->Draw();

      c2->cd(i + 1);
      hFangCutsCos[i]->DrawCopy("ep same");
      if (kRunTGenPhaseSpace) {
         hGenbodCos[i]->DrawCopy("ep same");
      }
      leg[i + kNBody]->Draw();

      c3->cd(i + 1);
      hFangCutsPhi[i]->DrawCopy("ep same");
      if (kRunTGenPhaseSpace) {
         hGenbodPhi[i]->DrawCopy("ep same");
      }
      leg[i + 2 * kNBody]->Draw();
   }

   //==========================================================================
   // Test 3: Elastic ep Scattering Cross Section
   //==========================================================================
   std::cout << "\n========================================" << std::endl;
   std::cout << "Test 3: Elastic ep Differential Cross Section" << std::endl;
   std::cout << "========================================" << std::endl;

   const Int_t kNBody2 = 2;
   const Int_t kNDet2 = 1;
   nLoop = 1E5;
   nEvents = 0;

   Double_t massElectron = 0.000511;      // GeV
   Double_t massProton = 0.938272029;     // proton mass in GeV

   Double_t omega2[kNDet2];
   omega2[0] = 0.0;
   Double_t shape2[kNDet2];

   // Setup kinematics
   ROOT::Math::PxPyPzMVector pTarget(0.0, 0.0, 0.0, massProton);
   Double_t kineticE = 3.0;  // GeV electron kinetic energy
   Double_t gamma = kineticE / massElectron + 1.0;
   Double_t beta = TMath::Sqrt(1.0 - 1.0 / (gamma * gamma));
   ROOT::Math::PxPyPzMVector pBeam(0.0, 0.0, gamma * beta * massElectron, massElectron);
   ROOT::Math::PxPyPzMVector pTotal2 = pBeam + pTarget;

   Double_t masses2[kNBody2] = {massElectron, massProton};

   Double_t alphaQED = 1.0 / 137.0;
   Double_t ampSquared = 0.0;
   weight = 0.0;
   sumW = 0.0;
   sumW2 = 0.0;

   ROOT::Math::PxPyPzMVector pProtonIn, pElectronIn, pProtonOut, pElectronOut, pMomTransfer;
   Double_t lambda, tau, formGE, formGM, qSquared;

   pElectronIn = pBeam;
   pProtonIn = pTarget;
   Double_t flux = 1.0 / (16.0 * kPi * kPi *
                          TMath::Sqrt(pElectronIn.Dot(pProtonIn) * pElectronIn.Dot(pProtonIn) - 
                                      massElectron * massElectron * massProton * massProton));

   // Setup Rosenbluth function for comparison
   TF1 *fRosenbluth = new TF1("fRosenbluth", fElastic, -1, 0.9992, 1);
   Double_t parElastic[1] = {kineticE};
   fRosenbluth->SetParameters(parElastic);

   //==========================================================================
   // FANG Point Generation: Differential Cross Section at Specific Angles
   //==========================================================================
   Double_t sigmaArr[11];
   Double_t sigmaErrArr[11];
   Double_t cosThetaArr[11];
   Double_t cosThetaErrArr[11];
   Double_t cosTheta;

   std::cout << "\nCalculating differential cross section at specific angles:" << std::endl;

   for (Int_t l = 0; l < 11; l++) {
      ampSquared = 0.0;
      weight = 0.0;
      sumW = 0.0;
      sumW2 = 0.0;
      nEvents = 0;
      v3Det.clear();

      cosTheta = -0.99 + l * 0.2;
      if (l == 10) cosTheta = 0.95;
      cosThetaArr[l] = cosTheta;
      cosThetaErrArr[l] = 0.0;

      v3.SetXYZ(TMath::Sqrt(1.0 - cosTheta * cosTheta), 0.0, cosTheta);
      v3Det.push_back(v3);
      shape2[0] = kModePoint;  // Point generation

      for (Int_t k = 0; k < nLoop; k++) {
         vecVecP.clear();
         vecWi.clear();
         eventStatus = GenFANG(kNBody2, pTotal2, masses2, omega2, shape2, v3Det, vecVecP, vecWi, &rng);
         if (!eventStatus) continue;

         for (size_t i = 0; i < vecVecP.size(); i++) {
            vecP = vecVecP[i];
            weight = vecWi[i];
            pElectronOut = vecP[0];
            pProtonOut = vecP[1];
            pMomTransfer = pElectronIn - pElectronOut;
            ROOT::Math::PxPyPzMVector pU = pTarget - pElectronOut;
            qSquared = -pMomTransfer.M2();
            formGE = 1.0 / ((1.0 + qSquared / kDipoleMassSq) *
                            (1.0 + qSquared / kDipoleMassSq));
            formGM = kProtonMagneticMoment * formGE;
            tau = qSquared / (4.0 * massProton * massProton);
            lambda = (pTotal2.M2() - pU.M2()) / (4.0 * massProton * massProton);

            // Calculate squared amplitude
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

      sigmaArr[l] = flux * sumW / nEvents;
      sigmaErrArr[l] = flux * TMath::Sqrt(sumW2) / nEvents;

      std::cout << "  cos(theta) = " << cosTheta
                << ": dsigma/dOmega = " << sigmaArr[l] << " +/- " << sigmaErrArr[l]
                << " (FANG/Rosenbluth = " << sigmaArr[l] / fRosenbluth->Eval(cosTheta) << ")"
                << std::endl;
   }

   TGraphErrors *grElastic = new TGraphErrors(11, cosThetaArr, sigmaArr, cosThetaErrArr, sigmaErrArr);
   grElastic->SetMarkerStyle(20);
   grElastic->SetMarkerSize(1.3);

   //==========================================================================
   // FANG Event Generation: Full Angular Distribution
   //==========================================================================
   std::cout << "\nGenerating full angular distribution..." << std::endl;

   Double_t sinTheta, phi, r1;

   TH1D *hXsec = new TH1D("hXsec", "hXsec", 440, -1.1, 1.1);
   TH1D *hNorm = new TH1D("hNorm", "hNorm", 440, -1.1, 1.1);
   TH1D *hCount = new TH1D("hCount", "hCount", 440, -1.1, 1.1);
   TH1D *hError = new TH1D("hError", "hError", 440, -1.1, 1.1);
   hXsec->SetMinimum(1E-17);
   hNorm->SetMinimum(1E-17);
   hError->SetMinimum(0.999);

   // Generate events in multiple r1 ranges to cover full angular range
   // Using 1/r^2 distribution to importance-sample forward angles

   struct Range_t {
      Double_t fC1;
      Double_t fC2;
   };
   Range_t ranges[4] = {{1.0, 2.0}, {0.4, 1.0}, {0.12, 0.4}, {0.01, 0.12}};

   for (Int_t rangeIdx = 0; rangeIdx < 4; rangeIdx++) {
      Double_t c1 = ranges[rangeIdx].fC1;
      Double_t c2 = ranges[rangeIdx].fC2;

      std::cout << "  Range " << rangeIdx + 1 << ": cos(theta) in ["
                << 1.0 - c2 << ", " << 1.0 - c1 << "]" << std::endl;

      for (Int_t k = 0; k < 1000000; k++) {
         // Generate r1 with 1/r^2 distribution
         r1 = c1 * c1 * c2 / (c2 * c1 - rng.Uniform(0, 1) * c1 * (c2 - c1));
         cosTheta = 1.0 - r1;
         sinTheta = TMath::Sqrt(1.0 - cosTheta * cosTheta);
         phi = rng.Uniform(0, kTwoPi);

         v3Det.clear();
         v3.SetXYZ(sinTheta * TMath::Cos(phi), sinTheta * TMath::Sin(phi), cosTheta);
         v3Det.push_back(v3);

         vecVecP.clear();
         vecWi.clear();
         eventStatus = GenFANG(kNBody2, pTotal2, masses2, omega2, shape2, v3Det, vecVecP, vecWi, &rng);
         if (!eventStatus) continue;

         for (size_t i = 0; i < vecVecP.size(); i++) {
            vecP = vecVecP[i];
            weight = vecWi[i];
            pElectronOut = vecP[0];
            pProtonOut = vecP[1];
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

            // Reweight from 1/r^2 to flat distribution
            Double_t reweight = r1 * r1 * (c2 - c1) / c1 / c2;
            hXsec->Fill(TMath::Cos(pElectronOut.Theta()), reweight * weight);
            hNorm->Fill(TMath::Cos(pElectronOut.Theta()), reweight);
            hCount->Fill(TMath::Cos(pElectronOut.Theta()), 1.0);
         }
      }
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

   // Configure histogram appearance
   hXsec->SetYTitle("#frac{d#sigma}{d#Omega}(ep -> ep)    [GeV^{-2}]");
   hXsec->SetXTitle("cos(#theta_{LAB})");
   hXsec->SetTitle("Electron Energy E=3 GeV");

   // Create final comparison plot
   TLegend *legFinal = new TLegend(0.12, 0.68, 0.42, 0.88);

   TCanvas *cFinal = new TCanvas("cFinal", "cFinal his", 10, 10, 1800, 1500);
   gPad->SetLogy();
   TH1F *vFrame = gPad->DrawFrame(-1.2, 5E-10, 1, 5E-3);
   hXsec->Draw("hist same");
   grElastic->Draw("P");
   fRosenbluth->Draw("same");

   legFinal->AddEntry(hXsec, "FANG event generation", "l");
   legFinal->AddEntry(grElastic, "FANG point calculation", "p");
   legFinal->AddEntry(fRosenbluth, "Rosenbluth cross section", "l");
   legFinal->Draw();

   vFrame->SetYTitle("#frac{d#sigma}{d#Omega}(ep -> ep)    [GeV^{-2}]");
   vFrame->SetXTitle("cos(#theta_{LAB})");
   vFrame->SetTitle("Electron Energy E=3 GeV");

   // Additional diagnostic plots
   TCanvas *cDiag = new TCanvas("cDiag", "cDiag Wi error", 10, 10, 1800, 1500);
   cDiag->Divide(2, 1);
   cDiag->cd(1);
   hNorm->Draw("hist");
   cDiag->cd(2);
   hCount->Draw("hist");

   std::cout << "\n========================================" << std::endl;
   std::cout << "runFANG() completed successfully" << std::endl;
   std::cout << "J. High Energ. Phys. 2025, 137 (2025). https://doi.org/10.1007/JHEP12(2025)137" << std::endl;
    std::cout << "========================================" << std::endl;
}
