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
/// \file TFANG.cxx
///
/// \brief Implementation of TFANG (Focused Angular N-body event Generator)
/// \authors Arik Kreisel, Itay Horin
///
/// TFANG is a Monte Carlo tool for efficient event generation in restricted
/// (or full) Lorentz-Invariant Phase Space (LIPS). Unlike conventional approaches
/// that always sample the full 4pi solid angle, TFANG can also directly generate
/// events in which selected final-state particles are constrained to fixed
/// directions or finite angular regions in the laboratory frame.
///
/// Reference: Horin, I., Kreisel, A. & Alon, O. Focused angular N -body event generator (FANG).
/// J. High Energ. Phys. 2025, 137 (2025). 
/// https://doi.org/10.1007/JHEP12(2025)137 
/// https://arxiv.org/abs/2509.11105 
////////////////////////////////////////////////////////////////////////////////


#include "TFANG.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TError.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/Boost.h"
#include "Math/GenVector/Polar3D.h"

#include <cmath>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
// FANG namespace - implementation
////////////////////////////////////////////////////////////////////////////////
namespace FANG {

////////////////////////////////////////////////////////////////////////////////
/// \brief Node_t constructor implementation
////////////////////////////////////////////////////////////////////////////////
Node_t::Node_t(const ROOT::Math::PxPyPzMVector &p1,
               const ROOT::Math::PxPyPzMVector &p2,
               Double_t weight, Node_t *parent)
   : fPV(p2)
   , fPDet(p1)
   , fWeight(weight)
   , fLeft(nullptr)
   , fRight(nullptr)
   , fParent(parent)
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Recursively delete a tree and free all memory
////////////////////////////////////////////////////////////////////////////////
void DeleteTree(Node_t *node)
{
   if (node == nullptr)
      return;
   DeleteTree(node->fLeft);
   DeleteTree(node->fRight);
   delete node;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Create the first (root) node of the tree
////////////////////////////////////////////////////////////////////////////////
Node_t *CreateFirst(Node_t *node,
                    const ROOT::Math::PxPyPzMVector &p1,
                    const ROOT::Math::PxPyPzMVector &p2,
                    Double_t weight)
{
   if (node == nullptr) {
      return new Node_t(p1, p2, weight, nullptr);
   }
   return node;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Create a right child node
////////////////////////////////////////////////////////////////////////////////
Node_t *CreateRight(Node_t *node, Node_t *tmp,
                    const ROOT::Math::PxPyPzMVector &p1,
                    const ROOT::Math::PxPyPzMVector &p2,
                    Double_t weight)
{
   if (node == nullptr) {
      return new Node_t(p1, p2, weight, tmp);
   }
   node->fRight = CreateRight(node->fRight, node, p1, p2, weight);
   return node;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Create a left child node
////////////////////////////////////////////////////////////////////////////////
Node_t *CreateLeft(Node_t *node, Node_t *tmp,
                   const ROOT::Math::PxPyPzMVector &p1,
                   const ROOT::Math::PxPyPzMVector &p2,
                   Double_t weight)
{
   if (node == nullptr) {
      return new Node_t(p1, p2, weight, tmp);
   }
   node->fLeft = CreateLeft(node->fLeft, node, p1, p2, weight);
   return node;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Collect all root-to-leaf paths for 4-momenta
////////////////////////////////////////////////////////////////////////////////
void CollectPaths(Int_t nBody, Node_t *node,
                  std::vector<ROOT::Math::PxPyPzMVector> &path,
                  std::vector<std::vector<ROOT::Math::PxPyPzMVector>> &paths)
{
   if (node == nullptr)
      return;

   path.push_back(node->fPDet);

   if (node->fLeft == nullptr && node->fRight == nullptr &&
       path.size() == static_cast<size_t>(nBody + 1)) {
      paths.push_back(path);
   } else {
      CollectPaths(nBody, node->fLeft, path, paths);
      CollectPaths(nBody, node->fRight, path, paths);
   }

   path.pop_back();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Collect all root-to-leaf paths for weights
////////////////////////////////////////////////////////////////////////////////
void CollectPathsWeights(Int_t nBody, Node_t *node,
                         std::vector<Double_t> &path,
                         std::vector<std::vector<Double_t>> &paths)
{
   if (node == nullptr)
      return;

   path.push_back(node->fWeight);

   if (node->fLeft == nullptr && node->fRight == nullptr &&
       path.size() == static_cast<size_t>(nBody + 1)) {
      paths.push_back(path);
   } else {
      CollectPathsWeights(nBody, node->fLeft, path, paths);
      CollectPathsWeights(nBody, node->fRight, path, paths);
   }

   path.pop_back();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Phase space kinematic function F(x,y) = sqrt((1-x-y)^2 - 4xy)
////////////////////////////////////////////////////////////////////////////////
Double_t CalcKMFactor(Double_t x, Double_t y)
{
   Double_t arg = (1.0 - x - y) * (1.0 - x - y) - 4.0 * x * y;
   if (arg < 0) {
      ::Warning("TFANG::CalcKMFactor", "Received negative sqrt argument: %g", arg);
      return 0.0;
   }
   return std::sqrt(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Generate isotropic two-body decay
////////////////////////////////////////////////////////////////////////////////
void TwoBody(const ROOT::Math::PxPyPzMVector &S,
             Double_t m1, Double_t m2,
             ROOT::Math::PxPyPzMVector &p1,
             ROOT::Math::PxPyPzMVector &p2,
             TRandom3 *rng)
{
   Double_t cst = rng->Uniform(-1.0, 1.0);
   Double_t snt = std::sqrt(1.0 - cst * cst);
   Double_t phi = rng->Uniform(0.0, kTwoPi);

   Double_t E1 = (S.M2() - m2 * m2 + m1 * m1) / (2.0 * S.M());

   if ((E1 * E1 - m1 * m1) < 0) {
      ::Error("TFANG::TwoBody", "E1^2 - m1^2 < 0, E1=%g, m1=%g", E1, m1);
      return;
   }

   Double_t sp = std::sqrt(E1 * E1 - m1 * m1);

   ROOT::Math::PxPyPzMVector p1CM(sp * snt * std::cos(phi),
                                   sp * snt * std::sin(phi),
                                   sp * cst, m1);
   ROOT::Math::PxPyPzMVector p2CM(-sp * snt * std::cos(phi),
                                   -sp * snt * std::sin(phi),
                                   -sp * cst, m2);

   ROOT::Math::XYZVector betaVS = S.BoostToCM();
   ROOT::Math::Boost bstCM;
   bstCM.SetComponents(betaVS);
   ROOT::Math::Boost bstLAB = bstCM.Inverse();

   p1 = bstLAB(p1CM);
   p1.SetM(m1);
   p2 = bstLAB(p2CM);
   p2.SetM(m2);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate 4-momentum for particle constrained to a lab-frame direction
////////////////////////////////////////////////////////////////////////////////
Bool_t TGenPointSpace(const ROOT::Math::PxPyPzMVector &S1,
                      Double_t m1, Double_t m2,
                      ROOT::Math::XYZVector vDet,
                      Int_t &solutions,
                      Double_t *jackPDF,
                      ROOT::Math::PxPyPzMVector *pDet,
                      ROOT::Math::PxPyPzMVector *pD2)
{
   ROOT::Math::XYZVector VSu(S1.Px() / S1.P(),
                              S1.Py() / S1.P(),
                              S1.Pz() / S1.P());
   VSu = VSu.Unit();

   ROOT::Math::XYZVector betaVS(-S1.Beta() * VSu.X(),
                                 -S1.Beta() * VSu.Y(),
                                 -S1.Beta() * VSu.Z());
   ROOT::Math::Boost bstCM;
   bstCM.SetComponents(betaVS);
   ROOT::Math::Boost bstLAB = bstCM.Inverse();

   vDet = vDet.Unit();
   LongDouble_t cosLAB = VSu.Dot(vDet);
   LongDouble_t sinLAB = std::sqrt(1.0 - cosLAB * cosLAB);

   LongDouble_t mCM = S1.M();
   LongDouble_t ECM = S1.E();
   LongDouble_t pCM = S1.P();
   LongDouble_t gamma1 = S1.Gamma();

   LongDouble_t CME3 = (mCM * mCM - m2 * m2 + m1 * m1) / (2.0 * mCM);

   if ((CME3 * CME3 - m1 * m1) < 0) {
      ::Error("TFANG::TGenPointSpace", "CMp3 negative sqrt");
      ::Error("TFANG::TGenPointSpace", "S1.M()=%g S1.P()=%g", (Double_t)S1.M(), (Double_t)S1.P());
      ::Error("TFANG::TGenPointSpace", "m1=%g m2=%g", m1, m2);
      solutions = 0;
      return kFALSE;
   }

   LongDouble_t CMp3 = std::sqrt(CME3 * CME3 - m1 * m1);

   LongDouble_t aa = pCM * pCM * cosLAB * cosLAB - ECM * ECM;
   LongDouble_t bb = 2.0 * pCM * cosLAB * CME3 * mCM;
   LongDouble_t cc = CME3 * mCM * CME3 * mCM - m1 * m1 * ECM * ECM;

   LongDouble_t discriminant = bb * bb - 4.0 * aa * cc;

   jackPDF[0] = 0.0;
   jackPDF[1] = 0.0;
   solutions = 1;

   if (discriminant < 0) {
      solutions = 0;
      return kFALSE;
   }

   LongDouble_t p3LAB[2];
   LongDouble_t sqrtDisc = std::sqrt(discriminant);

   p3LAB[0] = (-bb + sqrtDisc) / (2.0 * aa);

   if (p3LAB[0] <= 0) {
      p3LAB[0] = (-bb - sqrtDisc) / (2.0 * aa);
      if (p3LAB[0] <= 0) {
         solutions = 0;
         return kFALSE;
      }
   } else {
      p3LAB[1] = (-bb - sqrtDisc) / (2.0 * aa);
      if (p3LAB[1] > 0) {
         solutions = 2;
      }
   }

   LongDouble_t pdfCM = 1.0 / kFourPi;

   for (Int_t l = 0; l < solutions; l++) {
      pDet[l].SetCoordinates(p3LAB[l] * vDet.X(),
                             p3LAB[l] * vDet.Y(),
                             p3LAB[l] * vDet.Z(), m1);

      ROOT::Math::PxPyPzMVector p3CM = bstCM(pDet[l]);
      p3CM.SetM(m1);

      ROOT::Math::PxPyPzMVector p4CM(-p3CM.Px(), -p3CM.Py(), -p3CM.Pz(), m2);
      pD2[l] = bstLAB(p4CM);
      pD2[l].SetM(m2);

      if (std::abs(pD2[l].M() - m2) > kMomentumTolerance) {
         ::Warning("TFANG::TGenPointSpace", "Mass mismatch: %g != %g",
                   pD2[l].M(), m2);
      }

      LongDouble_t cosCM = p3CM.Vect().Dot(VSu) / CMp3;
      LongDouble_t qqq = pCM * p3CM.E() / (ECM * p3CM.P());

      LongDouble_t Jack;

      if (std::abs(cosLAB) > 0.99 && std::abs(cosCM) > 0.99) {
         Jack = gamma1 * gamma1 * cosCM * cosCM *
                (1.0 + qqq * cosCM) * (1.0 + qqq * cosCM) /
                (cosLAB * cosLAB);
      } else {
         Jack = ((1.0 - cosCM) * (1.0 + cosCM)) /
                ((1.0 - cosLAB) * (1.0 + cosLAB)) *
                std::sqrt(((1.0 - cosCM) * (1.0 + cosCM)) /
                         ((1.0 - cosLAB) * (1.0 + cosLAB))) /
                gamma1 / (1.0 + qqq * cosCM);
      }

      jackPDF[l] = std::abs(pdfCM * Jack);
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Generate random direction vector within specified solid angle
////////////////////////////////////////////////////////////////////////////////
void TGenVec(Double_t Omega, Double_t Ratio,
             ROOT::Math::XYZVector Vcenter,
             ROOT::Math::XYZVector &vPoint,
             TRandom3 *rng)
{
   ROOT::Math::XYZVector newZ, newX, newY, Vz;
   ROOT::Math::Polar3DVector Vgen;
   Double_t cst, phi, Dphi, Dcos, phi0, cst0;

   if (Omega > kFourPi || Omega < 0) {
      Omega = kFourPi;
      ::Warning("TFANG::TGenVec", "Omega out of range, set to 4pi");
   }

   if (Ratio > 1.0) {
      Ratio = 0.0;
      ::Warning("TFANG::TGenVec", "Ratio out of range, set to 0");
   }

   if (IsCircle(Ratio)) {
      cst = rng->Uniform(1.0 - Omega / kTwoPi, 1.0);
      phi = rng->Uniform(0.0, kTwoPi);

      if (std::abs(Vcenter.X()) < kPositionTolerance &&
          std::abs(Vcenter.Y()) < kPositionTolerance) {
         if (Vcenter.Z() > 0) {
            Vgen.SetCoordinates(1.0, std::acos(cst), phi);
         } else {
            Vgen.SetCoordinates(1.0, std::acos(-cst), phi);
         }
         vPoint = Vgen;
      } else {
         Vz.SetXYZ(0, 0, 1);
         newZ = Vcenter.Unit();
         newY = newZ.Cross(Vz).Unit();
         newX = newY.Cross(newZ).Unit();
         ROOT::Math::Rotation3D m(newX.X(), newY.X(), newZ.X(),
                                   newX.Y(), newY.Y(), newZ.Y(),
                                   newX.Z(), newY.Z(), newZ.Z());
         Vgen.SetCoordinates(1.0, std::acos(cst), phi);
         vPoint = m * Vgen;
      }
   } else if (IsStrip(Ratio)) {
      Dphi = Ratio * kTwoPi;
      Dcos = Omega / Dphi;
      phi0 = Vcenter.Phi();
      cst0 = std::cos(Vcenter.Theta());

      if (cst0 > (1.0 - Dcos / 2.0)) {
         cst0 = 1.0 - Dcos / 2.0;
         ::Warning("TFANG::TGenVec", "Center moved to agree with Omega (near +1)");
      }
      if (cst0 < (-1.0 + Dcos / 2.0)) {
         cst0 = -1.0 + Dcos / 2.0;
         ::Warning("TFANG::TGenVec", "Center moved to agree with Omega (near -1)");
      }

      cst = rng->Uniform(cst0 - Dcos / 2.0, cst0 + Dcos / 2.0);
      phi = rng->Uniform(phi0 - Dphi / 2.0, phi0 + Dphi / 2.0);
      Vgen.SetCoordinates(1.0, std::acos(cst), phi);
      vPoint = Vgen;
   } else if (IsRing(Ratio)) {
      cst = 1.0 - Omega / kTwoPi;
      phi = rng->Uniform(0.0, kTwoPi);

      if (std::abs(Vcenter.X()) < kPositionTolerance &&
          std::abs(Vcenter.Y()) < kPositionTolerance) {
         if (Vcenter.Z() > 0) {
            Vgen.SetCoordinates(1.0, std::acos(cst), phi);
         } else {
            Vgen.SetCoordinates(1.0, std::acos(-cst), phi);
         }
         vPoint = Vgen;
      } else {
         Vz.SetXYZ(0, 0, 1);
         newZ = Vcenter.Unit();
         newY = newZ.Cross(Vz).Unit();
         newX = newY.Cross(newZ).Unit();
         ROOT::Math::Rotation3D m(newX.X(), newY.X(), newZ.X(),
                                   newX.Y(), newY.Y(), newZ.Y(),
                                   newX.Z(), newY.Z(), newZ.Z());
         Vgen.SetCoordinates(1.0, std::acos(cst), phi);
         vPoint = m * Vgen;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Main FANG generator function
////////////////////////////////////////////////////////////////////////////////
Int_t GenFANG(Int_t nBody,
              const ROOT::Math::PxPyPzMVector &S,
              const Double_t *masses,
              const Double_t *Om,
              const Double_t *Ratio,
              std::vector<ROOT::Math::XYZVector> V3Det,
              std::vector<std::vector<ROOT::Math::PxPyPzMVector>> &VecVecP,
              std::vector<Double_t> &vecWi,
              TRandom3 *rng)
{
   Int_t nDet = static_cast<Int_t>(V3Det.size());
   Double_t mS = S.M();
   Double_t wh = 1.0;
   Double_t mB, mA, mall, whPS;

   mall = 0.0;
   for (Int_t l = 0; l < nBody; l++) {
      mall += masses[l];
   }

   if (mall >= mS) {
      ::Error("TFANG::GenFANG", "Sum of decay masses (%g) >= parent mass (%g)",
              mall, mS);
      return 0;
   }

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> pathsP;
   std::vector<std::vector<Double_t>> pathsJ;
   std::vector<ROOT::Math::PxPyPzMVector> vecP;
   std::vector<Double_t> vecJ;
   std::vector<Int_t> branch;

   vecP.clear();
   vecJ.clear();
   vecWi.clear();
   pathsJ.clear();

   Bool_t Hit;
   ROOT::Math::XYZVector V3;
   ROOT::Math::PxPyPzMVector p1;
   ROOT::Math::PxPyPzMVector p2;
   ROOT::Math::PxPyPzMVector pV;
   Int_t solutions;
   Double_t jackPDF[2];
   ROOT::Math::PxPyPzMVector pDet[2];
   ROOT::Math::PxPyPzMVector pD2[2];

   std::vector<Double_t> mV(nBody - 2);
   std::vector<Double_t> rrr(nBody - 2);

   //==========================================================================
   // Two-body decay case
   //==========================================================================
   if (nBody == 2) {
      whPS = CalcKMFactor(masses[0] * masses[0] / S.M2(),
                          masses[1] * masses[1] / S.M2()) * kPi / 2.0;

      if (nDet == 1) {
         if (IsPoint(Ratio[0])) {
            V3 = V3Det[0].Unit();
         } else {
            TGenVec(Om[0], Ratio[0], V3Det[0].Unit(), V3, rng);
         }

         Hit = TGenPointSpace(S, masses[0], masses[1], V3, solutions,
                              jackPDF, pDet, pD2);
         if (!Hit)
            return 0;

         for (Int_t l = 0; l < solutions; l++) {
            vecP.clear();
            vecP.push_back(pDet[l]);
            vecP.push_back(pD2[l]);
            VecVecP.push_back(vecP);
            vecWi.push_back(jackPDF[l] * whPS);
         }
      } else {
         TwoBody(S, masses[0], masses[1], p1, p2, rng);
         vecP.push_back(p1);
         vecP.push_back(p2);
         wh = 1.0;
         vecWi.push_back(wh * whPS);
         VecVecP.push_back(vecP);
      }

      return 1;
   }

   //==========================================================================
   // N-body decay case (nBody > 2)
   //==========================================================================

   rng->RndmArray(nBody - 2, rrr.data());
   std::sort(rrr.begin(), rrr.end());

   for (Int_t i = 0; i < nBody - 2; i++) {
      mB = 0.0;
      mA = 0.0;
      for (Int_t l = 0; l < i + 1; l++) {
         mB += masses[l];
      }
      for (Int_t l = i + 1; l < nBody; l++) {
         mA += masses[l];
      }
      mV[i] = rrr[nBody - 2 - i - 1] * (mS - mall) + mA;

      if (i > 0 && (mV[i - 1] - mV[i]) < masses[i]) {
         ::Error("TFANG::GenFANG",
                 "Virtual mass constraint violated at i=%d, mV[i-1]=%g, mV[i]=%g, masses[i]=%g",
                 i, mV[i - 1], mV[i], masses[i]);
      }
   }

   whPS = mV[0] * CalcKMFactor(masses[0] * masses[0] / S.M2(),
                               mV[0] * mV[0] / S.M2());

   for (Int_t i = 0; i < nBody - 3; i++) {
      whPS *= mV[i + 1] * CalcKMFactor(masses[i + 1] * masses[i + 1] / (mV[i] * mV[i]),
                                       mV[i + 1] * mV[i + 1] / (mV[i] * mV[i]));
   }

   if (nBody > 2) {
      whPS *= CalcKMFactor(masses[nBody - 2] * masses[nBody - 2] / (mV[nBody - 3] * mV[nBody - 3]),
                           masses[nBody - 1] * masses[nBody - 1] / (mV[nBody - 3] * mV[nBody - 3]));
   }

   whPS *= std::pow(kPi, nBody - 1) / 2.0 *
           std::pow(mS - mall, nBody - 2) / TMath::Factorial(nBody - 2);

   //==========================================================================
   // No detector constraints
   //==========================================================================
   if (nDet == 0) {
      TwoBody(S, masses[0], mV[0], p1, p2, rng);
      vecP.push_back(p1);
      pV = p2;

      for (Int_t i = 0; i < nBody - 3; i++) {
         TwoBody(pV, masses[i + 1], mV[i + 1], p1, p2, rng);
         vecP.push_back(p1);
         pV = p2;
      }

      TwoBody(pV, masses[nBody - 2], masses[nBody - 1], p1, p2, rng);
      vecP.push_back(p1);
      vecP.push_back(p2);

      wh = 1.0;
      vecWi.push_back(wh * whPS);
      VecVecP.push_back(vecP);

      return 1;
   }

   //==========================================================================
   // With detector constraints - use tree to track solutions
   //==========================================================================
   Node_t *root = CreateFirst(nullptr, S, S, 1.0);
   Node_t *cur = root;
   Int_t level = 0;
   branch.clear();

   while (level < (nBody - 1)) {

      // Case 1: Constrained particle, not the last two-body decay
      if (level < nDet && level < (nBody - 2)) {
         pV = cur->fPV;

         if (IsPoint(Ratio[level])) {
            V3 = V3Det[level].Unit();
         } else {
            TGenVec(Om[level], Ratio[level], V3Det[level].Unit(), V3, rng);
         }

         Hit = TGenPointSpace(pV, masses[level], mV[level], V3,
                              solutions, jackPDF, pDet, pD2);

         if (solutions == 0 && branch.empty()) {
            level = nBody - 1;
            continue;
         }
         if (solutions == 0 && !branch.empty()) {
            while (level > branch.back()) {
               level--;
               cur = cur->fParent;
            }
            cur = cur->fLeft;
            level++;
            branch.pop_back();
            continue;
         }
         if (solutions == 1) {
            CreateRight(cur, nullptr, pDet[0], pD2[0], jackPDF[0]);
            cur = cur->fRight;
            level++;
            continue;
         } else if (solutions == 2) {
            branch.push_back(level);
            CreateLeft(cur, nullptr, pDet[1], pD2[1], jackPDF[1]);
            CreateRight(cur, nullptr, pDet[0], pD2[0], jackPDF[0]);
            cur = cur->fRight;
            level++;
            continue;
         }
      }

      // Case 2: Constrained particle, last two-body decay
      if (level < nDet && level == (nBody - 2)) {
         pV = cur->fPV;

         if (IsPoint(Ratio[level])) {
            V3 = V3Det[level].Unit();
         } else {
            TGenVec(Om[level], Ratio[level], V3Det[level].Unit(), V3, rng);
         }

         Hit = TGenPointSpace(pV, masses[level], masses[level + 1], V3,
                              solutions, jackPDF, pDet, pD2);

         if (solutions == 0 && branch.empty()) {
            level = nBody - 1;
            continue;
         }
         if (solutions == 0 && !branch.empty()) {
            while (level > branch.back()) {
               level--;
               cur = cur->fParent;
            }
            cur = cur->fLeft;
            level++;
            branch.pop_back();
            continue;
         }
         if (solutions == 1) {
            CreateRight(cur, nullptr, pDet[0], pD2[0], jackPDF[0]);
            cur = cur->fRight;
            CreateRight(cur, nullptr, pD2[0], S, 1.0);
            cur = cur->fRight;
            level++;
            continue;
         } else if (solutions == 2) {
            CreateRight(cur, nullptr, pDet[0], pD2[0], jackPDF[0]);
            cur = cur->fRight;
            CreateRight(cur, nullptr, pD2[0], S, 1.0);
            cur = cur->fParent;
            CreateLeft(cur, nullptr, pDet[1], pD2[1], jackPDF[1]);
            cur = cur->fRight;
            CreateRight(cur, nullptr, pD2[1], S, 1.0);
            cur = cur->fRight;
            level++;

            if (level == (nBody - 1) && !branch.empty()) {
               while (level > branch.back()) {
                  level--;
                  cur = cur->fParent;
               }
               cur = cur->fParent;
               cur = cur->fLeft;
               level++;
               branch.pop_back();
               continue;
            }
            continue;
         }
      }

      // Case 3: Unconstrained particle, not the last two-body decay
      if (level >= nDet && level < nBody - 2) {
         pV = cur->fPV;
         TwoBody(pV, masses[level], mV[level], p1, p2, rng);
         CreateRight(cur, nullptr, p1, p2, 1.0);
         cur = cur->fRight;
         level++;
         continue;
      }

      // Case 4: Unconstrained particle, last two-body decay
      if (level >= nDet && level == nBody - 2) {
         pV = cur->fPV;
         TwoBody(pV, masses[level], masses[level + 1], p1, p2, rng);
         CreateRight(cur, nullptr, p1, p2, 1.0);
         cur = cur->fRight;
         CreateRight(cur, nullptr, p2, S, 1.0);
         cur = cur->fRight;
         level++;

         if (level == (nBody - 1) && !branch.empty()) {
            while (level > branch.back()) {
               level--;
               cur = cur->fParent;
            }
            cur = cur->fParent;
            cur = cur->fLeft;
            level++;
            branch.pop_back();
            continue;
         }
         continue;
      }

      // Backtrack if needed
      if (level == (nBody - 1) && !branch.empty()) {
         while (level > branch.back()) {
            level--;
            cur = cur->fParent;
         }
         cur = cur->fParent;
         cur = cur->fLeft;
         level++;
         branch.pop_back();
         continue;
      }

   } // end while

   CollectPathsWeights(nBody, root, vecJ, pathsJ);
   CollectPaths(nBody, root, vecP, pathsP);

   DeleteTree(root);
   root = nullptr;

   if (pathsP.empty()) {
      return 0;
   }

   for (size_t i = 0; i < pathsJ.size(); i++) {
      vecJ = pathsJ[i];
      vecP = pathsP[i];
      vecP.erase(vecP.begin());
      VecVecP.push_back(vecP);

      wh = 1.0;
      for (Int_t j = 1; j < nDet + 1; j++) {
         wh *= vecJ[j];
      }
      vecWi.push_back(wh * whPS);
   }

   return 1;
}

} // namespace FANG

////////////////////////////////////////////////////////////////////////////////
// TFANG class implementation
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
////////////////////////////////////////////////////////////////////////////////
TFANG::TFANG()
   : fNBody(0)
   , fS()
   , fMasses()
   , fOmega()
   , fRatio()
   , fV3Det()
   , fVecVecP()
   , fVecWi()
   , fRng(new TRandom3(0))
   , fOwnRng(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor with external random number generator
////////////////////////////////////////////////////////////////////////////////
TFANG::TFANG(TRandom3 *rng)
   : fNBody(0)
   , fS()
   , fMasses()
   , fOmega()
   , fRatio()
   , fV3Det()
   , fVecVecP()
   , fVecWi()
   , fRng(rng)
   , fOwnRng(kFALSE)
{
   if (fRng == nullptr) {
      fRng = new TRandom3(0);
      fOwnRng = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Destructor
////////////////////////////////////////////////////////////////////////////////
TFANG::~TFANG()
{
   if (fOwnRng && fRng != nullptr) {
      delete fRng;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Set decay configuration
////////////////////////////////////////////////////////////////////////////////
Bool_t TFANG::SetDecay(const ROOT::Math::PxPyPzMVector &S, Int_t nBody, const Double_t *masses)
{
   if (nBody < 2) {
      ::Error("TFANG::SetDecay", "nBody must be >= 2, got %d", nBody);
      return kFALSE;
   }

   if (masses == nullptr) {
      ::Error("TFANG::SetDecay", "masses array is null");
      return kFALSE;
   }

   Double_t totalMass = 0.0;
   for (Int_t i = 0; i < nBody; ++i) {
      if (masses[i] < 0.0) {
         ::Error("TFANG::SetDecay", "Negative mass at index %d: %g", i, masses[i]);
         return kFALSE;
      }
      totalMass += masses[i];
   }

   if (totalMass >= S.M()) {
      ::Error("TFANG::SetDecay", "Sum of decay masses (%g) >= parent mass (%g)",
              totalMass, S.M());
      return kFALSE;
   }

   fS = S;
   fNBody = nBody;
   fMasses.assign(masses, masses + nBody);

   fVecVecP.clear();
   fVecWi.clear();

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Add angular constraint for a particle
////////////////////////////////////////////////////////////////////////////////
void TFANG::AddConstraint(const ROOT::Math::XYZVector &direction, Double_t omega, Double_t ratio)
{
   if (fNBody == 0) {
      ::Warning("TFANG::AddConstraint", "SetDecay must be called before AddConstraint");
   }

   fV3Det.push_back(direction);
   fOmega.push_back(omega);
   fRatio.push_back(ratio);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Clear all constraints
////////////////////////////////////////////////////////////////////////////////
void TFANG::ClearConstraints()
{
   fV3Det.clear();
   fOmega.clear();
   fRatio.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Set random number generator seed
////////////////////////////////////////////////////////////////////////////////
void TFANG::SetSeed(UInt_t seed)
{
   if (fRng != nullptr) {
      fRng->SetSeed(seed);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Generate a single phase space event
////////////////////////////////////////////////////////////////////////////////
Int_t TFANG::Generate()
{
   if (fNBody < 2) {
      ::Error("TFANG::Generate", "SetDecay must be called before Generate");
      return 0;
   }

   fVecVecP.clear();
   fVecWi.clear();

   const Double_t *omPtr = fOmega.empty() ? nullptr : fOmega.data();
   const Double_t *ratioPtr = fRatio.empty() ? nullptr : fRatio.data();

   Int_t result = FANG::GenFANG(fNBody, fS, fMasses.data(),
                                omPtr, ratioPtr, fV3Det,
                                fVecVecP, fVecWi, fRng);

   if (result == 0) {
      return 0;
   }

   return static_cast<Int_t>(fVecVecP.size());
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Get weight (unconstrained mode)
////////////////////////////////////////////////////////////////////////////////
Double_t TFANG::GetWeight() const
{
   if (fVecWi.empty()) {
      ::Warning("TFANG::GetWeight", "No event generated yet");
      return 0.0;
   }
   return fVecWi[0];
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Get weight for specific solution
////////////////////////////////////////////////////////////////////////////////
Double_t TFANG::GetWeight(Int_t iSolution) const
{
   if (iSolution < 0 || iSolution >= static_cast<Int_t>(fVecWi.size())) {
      ::Error("TFANG::GetWeight", "Solution index %d out of range [0, %d)",
              iSolution, static_cast<Int_t>(fVecWi.size()));
      return 0.0;
   }
   return fVecWi[iSolution];
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Get 4-momentum of particle (single solution)
////////////////////////////////////////////////////////////////////////////////
ROOT::Math::PxPyPzMVector TFANG::GetDecay(Int_t iParticle) const
{
   return GetDecay(0, iParticle);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Get 4-momentum of particle for specific solution
////////////////////////////////////////////////////////////////////////////////
ROOT::Math::PxPyPzMVector TFANG::GetDecay(Int_t iSolution, Int_t iParticle) const
{
   if (iSolution < 0 || iSolution >= static_cast<Int_t>(fVecVecP.size())) {
      ::Error("TFANG::GetDecay", "Solution index %d out of range [0, %d)",
              iSolution, static_cast<Int_t>(fVecVecP.size()));
      return ROOT::Math::PxPyPzMVector();
   }

   const auto &vecP = fVecVecP[iSolution];
   if (iParticle < 0 || iParticle >= static_cast<Int_t>(vecP.size())) {
      ::Error("TFANG::GetDecay", "Particle index %d out of range [0, %d)",
              iParticle, static_cast<Int_t>(vecP.size()));
      return ROOT::Math::PxPyPzMVector();
   }

   return vecP[iParticle];
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Get all 4-momenta for a solution
////////////////////////////////////////////////////////////////////////////////
const std::vector<ROOT::Math::PxPyPzMVector> &TFANG::GetDecays(Int_t iSolution) const
{
   static const std::vector<ROOT::Math::PxPyPzMVector> empty;

   if (iSolution < 0 || iSolution >= static_cast<Int_t>(fVecVecP.size())) {
      ::Error("TFANG::GetDecays", "Solution index %d out of range [0, %d)",
              iSolution, static_cast<Int_t>(fVecVecP.size()));
      return empty;
   }

   return fVecVecP[iSolution];
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate phase space integral with uncertainty
////////////////////////////////////////////////////////////////////////////////
Bool_t TFANG::GetPhaseSpace(Long64_t nEvents, Double_t &phaseSpace, Double_t &error) const
{
   if (fNBody < 2) {
      ::Error("TFANG::GetPhaseSpace", "SetDecay must be called before GetPhaseSpace");
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   if (nEvents <= 0) {
      ::Error("TFANG::GetPhaseSpace", "nEvents must be positive, got %lld", nEvents);
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   TRandom3 localRng(fRng->GetSeed() + 1);

   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;
   Long64_t nSuccess = 0;

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   // Always run unconstrained - pass empty vectors for constraints
   std::vector<ROOT::Math::XYZVector> emptyV3Det;

   for (Long64_t i = 0; i < nEvents; ++i) {
      vecVecP.clear();
      vecWi.clear();

      Int_t result = FANG::GenFANG(fNBody, fS, fMasses.data(),
                                   nullptr, nullptr, emptyV3Det,
                                   vecVecP, vecWi, &localRng);

      if (result > 0) {
         for (size_t j = 0; j < vecWi.size(); ++j) {
            Double_t w = vecWi[j];
            sumW += w;
            sumW2 += w * w;
            ++nSuccess;
         }
      }
   }

   if (nSuccess == 0) {
      ::Warning("TFANG::GetPhaseSpace", "No successful events generated");
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   Double_t mean = sumW / static_cast<Double_t>(nSuccess);
   Double_t variance = (sumW2 / static_cast<Double_t>(nSuccess) - mean * mean);

   phaseSpace = mean;
   error = std::sqrt(variance / static_cast<Double_t>(nSuccess));

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate phase space with default 1E6 events
////////////////////////////////////////////////////////////////////////////////
Bool_t TFANG::GetPhaseSpace(Double_t &phaseSpace, Double_t &error) const
{
   return GetPhaseSpace(1000000, phaseSpace, error);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate partial (constrained) phase space integral with uncertainty
///
/// Runs Monte Carlo integration to estimate the partial phase space volume
/// when angular constraints are applied. The result is multiplied by the
/// product of all omega values (total solid angle factor).
////////////////////////////////////////////////////////////////////////////////
Bool_t TFANG::GetPartialPhaseSpace(Long64_t nEvents, Double_t &phaseSpace, Double_t &error) const
{
   if (fNBody < 2) {
      ::Error("TFANG::GetPartialPhaseSpace", "SetDecay must be called before GetPartialPhaseSpace");
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   if (fV3Det.empty()) {
      ::Error("TFANG::GetPartialPhaseSpace", "No constraints set. Use AddConstraint() first or use GetPhaseSpace() for unconstrained calculation");
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   if (nEvents <= 0) {
      ::Error("TFANG::GetPartialPhaseSpace", "nEvents must be positive, got %lld", nEvents);
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   // Calculate total solid angle factor based on constraint type
   Double_t totalOmega = 1.0;
   for (size_t i = 0; i < fOmega.size(); ++i) {
      if (FANG::IsPoint(fRatio[i])) {
         // Point constraint: factor of 1 (no change)
      } else if (FANG::IsRing(fRatio[i])) {
         // Ring constraint: factor of 2*pi
         totalOmega *= TMath::TwoPi();
      } else {
         // Cone constraint: use actual omega value
         totalOmega *= fOmega[i];
      }
   }

   TRandom3 localRng(fRng->GetSeed() + 1);

   Double_t sumW = 0.0;
   Double_t sumW2 = 0.0;
   Long64_t nSuccess = 0;

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> vecVecP;
   std::vector<Double_t> vecWi;

   const Double_t *omPtr = fOmega.data();
   const Double_t *ratioPtr = fRatio.data();

   for (Long64_t i = 0; i < nEvents; ++i) {
      vecVecP.clear();
      vecWi.clear();

      Int_t result = FANG::GenFANG(fNBody, fS, fMasses.data(),
                                   omPtr, ratioPtr, fV3Det,
                                   vecVecP, vecWi, &localRng);

      if (result > 0) {
         for (size_t j = 0; j < vecWi.size(); ++j) {
            Double_t w = vecWi[j];
            sumW += w;
            sumW2 += w * w;
            ++nSuccess;
         }
      }
   }

   if (nSuccess == 0) {
      ::Warning("TFANG::GetPartialPhaseSpace", "No successful events generated");
      phaseSpace = 0.0;
      error = 0.0;
      return kFALSE;
   }

   Double_t mean = sumW / static_cast<Double_t>(nSuccess);
   Double_t variance = (sumW2 / static_cast<Double_t>(nSuccess) - mean * mean);

   // Multiply by total solid angle factor
   phaseSpace = totalOmega * mean;
   error = totalOmega * std::sqrt(variance / static_cast<Double_t>(nSuccess));

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Calculate partial phase space with default 1E6 events
////////////////////////////////////////////////////////////////////////////////
Bool_t TFANG::GetPartialPhaseSpace(Double_t &phaseSpace, Double_t &error) const
{
   return GetPartialPhaseSpace(1000000, phaseSpace, error);
}
