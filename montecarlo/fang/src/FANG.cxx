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
/// \file FANG.cxx
/// \ingroup Physics
/// \brief Implementation of FANG (Focused Angular N-body event Generator)
/// \authors Arik Kreisel, Itay Horin
///
/// FANG is a Monte Carlo tool for efficient event generation in restricted
/// (or full) Lorentz-Invariant Phase Space (LIPS). Unlike conventional approaches
/// that always sample the full 4pi solid angle, FANG can also directly generate
/// events in which selected final-state particles are constrained to fixed
/// directions or finite angular regions in the laboratory frame.
///
/// Reference: Horin, I., Kreisel, A. & Alon, O. Focused angular N -body event generator (FANG).
/// J. High Energ. Phys. 2025, 137 (2025). 
/// https://doi.org/10.1007/JHEP12(2025)13 
/// https://arxiv.org/abs/2509.11105 
////////////////////////////////////////////////////////////////////////////////

#include "FANG.h"

#include "TRandom3.h"
#include "TMath.h"
#include "TError.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/Boost.h"
#include "Math/GenVector/Polar3D.h"

#include <cmath>
#include <algorithm>

namespace FANG {

////////////////////////////////////////////////////////////////////////////////
// Node_t Implementation
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Construct a new Node_t
/// \param[in] p1 Detected particle 4-momentum
/// \param[in] p2 Virtual system 4-momentum
/// \param[in] weight Weight value
/// \param[in] parent Pointer to parent node
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
// Tree Management Functions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Recursively delete a tree and free all memory
/// \param[in] node Root of the tree to delete
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
/// \param[in] node Existing node (should be nullptr for first call)
/// \param[in] p1 Detected particle 4-momentum
/// \param[in] p2 Virtual system 4-momentum
/// \param[in] weight Weight value
/// \return Pointer to created node
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
/// \param[in] node Current node
/// \param[in] tmp Parent node for new node
/// \param[in] p1 Detected particle 4-momentum
/// \param[in] p2 Virtual system 4-momentum
/// \param[in] weight Weight value
/// \return Pointer to node
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
/// \param[in] node Current node
/// \param[in] tmp Parent node for new node
/// \param[in] p1 Detected particle 4-momentum
/// \param[in] p2 Virtual system 4-momentum
/// \param[in] weight Weight value
/// \return Pointer to node
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
/// \param[in] nBody Number of bodies in the decay
/// \param[in] node Current node
/// \param[in,out] path Current path being built
/// \param[out] paths Output: all complete paths
////////////////////////////////////////////////////////////////////////////////
void CollectPaths(Int_t nBody, Node_t *node,
                  std::vector<ROOT::Math::PxPyPzMVector> &path,
                  std::vector<std::vector<ROOT::Math::PxPyPzMVector>> &paths)
{
   if (node == nullptr)
      return;

   path.push_back(node->fPDet);

   // If leaf node with correct path length, save the path
   if (node->fLeft == nullptr && node->fRight == nullptr &&
       path.size() == static_cast<size_t>(nBody + 1)) {
      paths.push_back(path);
   } else {
      CollectPaths(nBody, node->fLeft, path, paths);
      CollectPaths(nBody, node->fRight, path, paths);
   }

   path.pop_back(); // Backtrack
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Collect all root-to-leaf paths for weights
/// \param[in] nBody Number of bodies in the decay
/// \param[in] node Current node
/// \param[in,out] path Current path being built
/// \param[out] paths Output: all complete paths
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
// Utility Functions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Phase space kinematic function F(x,y) = sqrt((1-x-y)^2 - 4xy)
/// \param[in] x First mass ratio squared (m1^2/M^2)
/// \param[in] y Second mass ratio squared (m2^2/M^2)
/// \return Kinematic function value
////////////////////////////////////////////////////////////////////////////////
Double_t CalcKMFactor(Double_t x, Double_t y)
{
   Double_t arg = (1.0 - x - y) * (1.0 - x - y) - 4.0 * x * y;
   if (arg < 0) {
      ::Warning("FANG::CalcKMFactor", "Received negative sqrt argument: %g", arg);
      return 0.0;
   }
   return std::sqrt(arg);
}

////////////////////////////////////////////////////////////////////////////////
// Core Physics Functions
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Generate isotropic two-body decay
///
/// Performs a two-body decay isotropically in the rest frame of S,
/// then boosts results back to the lab frame.
///
/// \param[in] S Total 4-momentum of decaying system
/// \param[in] m1 Mass of first decay product
/// \param[in] m2 Mass of second decay product
/// \param[out] p1 4-momentum of first decay product (lab frame)
/// \param[out] p2 4-momentum of second decay product (lab frame)
/// \param[in] rng Pointer to TRandom3 random number generator (thread-safe)
////////////////////////////////////////////////////////////////////////////////
void TwoBody(const ROOT::Math::PxPyPzMVector &S,
             Double_t m1, Double_t m2,
             ROOT::Math::PxPyPzMVector &p1,
             ROOT::Math::PxPyPzMVector &p2,
             TRandom3 *rng)
{
   // Generate random direction in CM frame
   Double_t cst = rng->Uniform(-1.0, 1.0);
   Double_t snt = std::sqrt(1.0 - cst * cst);
   Double_t phi = rng->Uniform(0.0, kTwoPi);

   // Calculate energy and momentum in CM frame
   Double_t E1 = (S.M2() - m2 * m2 + m1 * m1) / (2.0 * S.M());

   if ((E1 * E1 - m1 * m1) < 0) {
      ::Error("FANG::TwoBody", "E1^2 - m1^2 < 0, E1=%g, m1=%g", E1, m1);
      return;
   }

   Double_t sp = std::sqrt(E1 * E1 - m1 * m1);

   // 4-momenta in CM frame
   ROOT::Math::PxPyPzMVector p1CM(sp * snt * std::cos(phi),
                                   sp * snt * std::sin(phi),
                                   sp * cst, m1);
   ROOT::Math::PxPyPzMVector p2CM(-sp * snt * std::cos(phi),
                                   -sp * snt * std::sin(phi),
                                   -sp * cst, m2);

   // Boost to lab frame
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
///
/// Given a two-body system S1 decaying to masses m1 and m2, with m1 constrained
/// to travel in direction vDet, calculate the possible 4-momenta.
///
/// \param[in] S1 Total 4-momentum of the decaying system
/// \param[in] m1 Mass of constrained particle
/// \param[in] m2 Mass of other particle
/// \param[in] vDet Unit vector specifying lab-frame direction for m1
/// \param[out] solutions Number of physical solutions (0, 1, or 2)
/// \param[out] jackPDF Array of Jacobian * PDF values for each solution
/// \param[out] pDet Array of 4-momenta for constrained particle
/// \param[out] pD2 Array of 4-momenta for other particle
/// \return kTRUE if at least one physical solution exists
////////////////////////////////////////////////////////////////////////////////
Bool_t TGenPointSpace(const ROOT::Math::PxPyPzMVector &S1,
                      Double_t m1, Double_t m2,
                      ROOT::Math::XYZVector vDet,
                      Int_t &solutions,
                      Double_t *jackPDF,
                      ROOT::Math::PxPyPzMVector *pDet,
                      ROOT::Math::PxPyPzMVector *pD2)
{
   // Direction of CM system in lab
   ROOT::Math::XYZVector VSu(S1.Px() / S1.P(),
                              S1.Py() / S1.P(),
                              S1.Pz() / S1.P());
   VSu = VSu.Unit();

   // Setup boost transformations
   ROOT::Math::XYZVector betaVS(-S1.Beta() * VSu.X(),
                                 -S1.Beta() * VSu.Y(),
                                 -S1.Beta() * VSu.Z());
   ROOT::Math::Boost bstCM;
   bstCM.SetComponents(betaVS);
   ROOT::Math::Boost bstLAB = bstCM.Inverse();

   vDet = vDet.Unit();
   LongDouble_t cosLAB = VSu.Dot(vDet);
   LongDouble_t sinLAB = std::sqrt(1.0 - cosLAB * cosLAB);

   // CM frame quantities
   LongDouble_t mCM = S1.M();
   LongDouble_t ECM = S1.E();
   LongDouble_t pCM = S1.P();
   LongDouble_t gamma1 = S1.Gamma();

   // Energy and momentum of outgoing particle in CM
   LongDouble_t CME3 = (mCM * mCM - m2 * m2 + m1 * m1) / (2.0 * mCM);

   if ((CME3 * CME3 - m1 * m1) < 0) {
      ::Error("FANG::TGenPointSpace", "CMp3 negative sqrt");
      ::Error("FANG::TGenPointSpace", "S1.M()=%g S1.P()=%g", (Double_t)S1.M(), (Double_t)S1.P());
      ::Error("FANG::TGenPointSpace", "m1=%g m2=%g", m1, m2);
      solutions = 0;
      return kFALSE;
   }

   LongDouble_t CMp3 = std::sqrt(CME3 * CME3 - m1 * m1);

   // Quadratic equation coefficients for lab momentum
   LongDouble_t aa = pCM * pCM * cosLAB * cosLAB - ECM * ECM;
   LongDouble_t bb = 2.0 * pCM * cosLAB * CME3 * mCM;
   LongDouble_t cc = CME3 * mCM * CME3 * mCM - m1 * m1 * ECM * ECM;

   LongDouble_t discriminant = bb * bb - 4.0 * aa * cc;

   // Initialize outputs
   jackPDF[0] = 0.0;
   jackPDF[1] = 0.0;
   solutions = 1;

   if (discriminant < 0) {
      solutions = 0;
      return kFALSE;
   }

   LongDouble_t p3LAB[2];
   LongDouble_t sqrtDisc = std::sqrt(discriminant);

   // Find physical solutions (positive momentum)
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

   // Calculate 4-momenta and Jacobians for each solution
   LongDouble_t pdfCM = 1.0 / kFourPi;

   for (Int_t l = 0; l < solutions; l++) {
      // Construct lab frame 4-momentum
      pDet[l].SetCoordinates(p3LAB[l] * vDet.X(),
                             p3LAB[l] * vDet.Y(),
                             p3LAB[l] * vDet.Z(), m1);

      // Boost to CM frame
      ROOT::Math::PxPyPzMVector p3CM = bstCM(pDet[l]);
      p3CM.SetM(m1);

      // Calculate other particle's 4-momentum
      ROOT::Math::PxPyPzMVector p4CM(-p3CM.Px(), -p3CM.Py(), -p3CM.Pz(), m2);
      pD2[l] = bstLAB(p4CM);
      pD2[l].SetM(m2);

      if (std::abs(pD2[l].M() - m2) > kMomentumTolerance) {
         ::Warning("FANG::TGenPointSpace", "Mass mismatch: %g != %g",
                   pD2[l].M(), m2);
      }

      // Calculate Jacobian: d(cos theta*)/d(cos theta_lab)
      LongDouble_t cosCM = p3CM.Vect().Dot(VSu) / CMp3;
      LongDouble_t qqq = pCM * p3CM.E() / (ECM * p3CM.P());

      LongDouble_t Jack;

      // Use appropriate formula depending on angle regime
      if (std::abs(cosLAB) > 0.99 && std::abs(cosCM) > 0.99) {
         // Near forward/backward direction - use alternative formula
         Jack = gamma1 * gamma1 * cosCM * cosCM *
                (1.0 + qqq * cosCM) * (1.0 + qqq * cosCM) /
                (cosLAB * cosLAB);
      } else {
         // General case
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
///
/// \param[in] Omega Solid angle size [steradians]
/// \param[in] Ratio Shape parameter determining generation mode
/// \param[in] Vcenter Central direction vector
/// \param[out] vPoint Generated direction vector
/// \param[in] rng Pointer to TRandom3 random number generator (thread-safe)
////////////////////////////////////////////////////////////////////////////////
void TGenVec(Double_t Omega, Double_t Ratio,
             ROOT::Math::XYZVector Vcenter,
             ROOT::Math::XYZVector &vPoint,
             TRandom3 *rng)
{
   ROOT::Math::XYZVector newZ, newX, newY, Vz;
   ROOT::Math::Polar3DVector Vgen;
   Double_t cst, phi, Dphi, Dcos, phi0, cst0;

   // Validate Omega
   if (Omega > kFourPi || Omega < 0) {
      Omega = kFourPi;
      ::Warning("FANG::TGenVec", "Omega out of range, set to 4pi");
   }

   // Validate Ratio for strip mode
   if (Ratio > 1.0) {
      Ratio = 0.0;
      ::Warning("FANG::TGenVec", "Ratio out of range, set to 0");
   }

   if (IsCircle(Ratio)) {
      // Circle generation: uniform within cone
      cst = rng->Uniform(1.0 - Omega / kTwoPi, 1.0);
      phi = rng->Uniform(0.0, kTwoPi);

      if (std::abs(Vcenter.X()) < kPositionTolerance &&
          std::abs(Vcenter.Y()) < kPositionTolerance) {
         // Special case: center along z-axis
         if (Vcenter.Z() > 0) {
            Vgen.SetCoordinates(1.0, std::acos(cst), phi);
         } else {
            Vgen.SetCoordinates(1.0, std::acos(-cst), phi);
         }
         vPoint = Vgen;
      } else {
         // General case: rotate to center direction
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
      // Strip generation: rectangular angular region
      Dphi = Ratio * kTwoPi;
      Dcos = Omega / Dphi;
      phi0 = Vcenter.Phi();
      cst0 = std::cos(Vcenter.Theta());

      // Adjust center if near poles
      if (cst0 > (1.0 - Dcos / 2.0)) {
         cst0 = 1.0 - Dcos / 2.0;
         ::Warning("FANG::TGenVec", "Center moved to agree with Omega (near +1)");
      }
      if (cst0 < (-1.0 + Dcos / 2.0)) {
         cst0 = -1.0 + Dcos / 2.0;
         ::Warning("FANG::TGenVec", "Center moved to agree with Omega (near -1)");
      }

      cst = rng->Uniform(cst0 - Dcos / 2.0, cst0 + Dcos / 2.0);
      phi = rng->Uniform(phi0 - Dphi / 2.0, phi0 + Dphi / 2.0);
      Vgen.SetCoordinates(1.0, std::acos(cst), phi);
      vPoint = Vgen;
   } else if (IsRing(Ratio)) {
      // Ring generation: fixed polar angle, random azimuthal
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
// Main Generator Function
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \brief Generate phase-space events with angular constraints
///
/// Main FANG generator function. Generates n-body phase space events
/// where selected particles are constrained to specified detector directions.
///
/// \param[in] nBody Number of outgoing particles
/// \param[in] S Total 4-momentum of the system
/// \param[in] masses Array of outgoing particle masses [GeV], length nBody
/// \param[in] Om Array of solid angles for constrained detectors [sr]
/// \param[in] Ratio Array of shape parameters for each detector:
///                  - = 2: Point generation (fixed direction)
///                  - = 0: Circle generation (uniform in cone)
///                  - 0 < Ratio[] <= 1: Strip generation (rectangular region)
///                              Dphi = Ratio[] * TwoPi;
///                              Dcos = Omega / Dphi;
///                  - < 0: Ring generation (fixed theta, uniform phi)
/// \param[in] V3Det Vector of direction vectors for constrained detectors
/// \param[out] VecVecP Output: vector of 4-momenta vectors for each solution
/// \param[out] vecWi Output: weight for each solution
/// \param[in] rng Pointer to TRandom3 random number generator (thread-safe)
/// \return 1 on success, 0 if no physical solution exists
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

   // Calculate total mass
   mall = 0.0;
   for (Int_t l = 0; l < nBody; l++) {
      mall += masses[l];
   }

   if (mall >= mS) {
      ::Error("FANG::GenFANG", "Sum of decay masses (%g) >= parent mass (%g)",
              mall, mS);
      return 0;
   }

   // Temporary storage
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

   // Virtual masses storage
   std::vector<Double_t> mV(nBody - 2);
   std::vector<Double_t> rrr(nBody - 2);

   //==========================================================================
   // Two-body decay case
   //==========================================================================
   if (nBody == 2) {
      whPS = CalcKMFactor(masses[0] * masses[0] / S.M2(),
                          masses[1] * masses[1] / S.M2()) * kPi / 2.0;

      if (nDet == 1) {
         // Constrained two-body decay
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
         // Unconstrained two-body decay (nDet == 0)
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

   // Generate virtual masses using M-generation algorithm
   rng->RndmArray(nBody - 2, rrr.data());

   // Sort random numbers in ascending order
   std::sort(rrr.begin(), rrr.end());

   // Calculate virtual masses
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
         ::Error("FANG::GenFANG",
                 "Virtual mass constraint violated at i=%d, mV[i-1]=%g, mV[i]=%g, masses[i]=%g",
                 i, mV[i - 1], mV[i], masses[i]);
      }
   }

   // Calculate phase space weight
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

   // Collect all paths from root to leaves
   CollectPathsWeights(nBody, root, vecJ, pathsJ);
   CollectPaths(nBody, root, vecP, pathsP);

   // Clean up tree memory
   DeleteTree(root);
   root = nullptr;

   if (pathsP.empty()) {
      return 0;
   }

   // Process all solutions
   for (size_t i = 0; i < pathsJ.size(); i++) {
      vecJ = pathsJ[i];
      vecP = pathsP[i];
      vecP.erase(vecP.begin()); // Remove first element (initial state)
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
