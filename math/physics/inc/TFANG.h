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
/// \file TFANG.h
///
/// \brief TFANG - Focused Angular N-body event Generator
/// \authors Arik Kreisel, Itay Horin
///
/// TFANG is a Monte Carlo tool for efficient event generation in restricted
/// (or full) Lorentz-Invariant Phase Space (LIPS). Unlike conventional approaches
/// that always sample the full 4pi solid angle, TFANG can also directly generate
/// events in which selected final-state particles are constrained to fixed
/// directions or finite angular regions in the laboratory frame.
///
/// This file contains:
/// - namespace FANG: Core generator functions and utilities
/// - class TFANG: User-friendly interface class
///
/// Reference: Horin, I., Kreisel, A. & Alon, O. Focused angular N -body event generator (FANG).
/// J. High Energ. Phys. 2025, 137 (2025). 
/// https://doi.org/10.1007/JHEP12(2025)137 
/// https://arxiv.org/abs/2509.11105 
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFANG
#define ROOT_TFANG

#include "Rtypes.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"

#include <vector>

class TRandom3;

////////////////////////////////////////////////////////////////////////////////
// FANG namespace - Core generator functions
////////////////////////////////////////////////////////////////////////////////
namespace FANG {

////////////////////////////////////////////////////////////////////////////////
/// Mathematical constants
////////////////////////////////////////////////////////////////////////////////
constexpr Double_t kPi     = 3.14159265358979323846;
constexpr Double_t kTwoPi  = 2.0 * kPi;
constexpr Double_t kFourPi = 4.0 * kPi;

////////////////////////////////////////////////////////////////////////////////
/// Physics constants
////////////////////////////////////////////////////////////////////////////////
constexpr Double_t kDipoleMassSq         = 0.71;    ///< GeV^2, for form factor
constexpr Double_t kProtonMagneticMoment = 2.793;   ///< mu_p in nuclear magnetons

////////////////////////////////////////////////////////////////////////////////
/// Numerical tolerances
////////////////////////////////////////////////////////////////////////////////
constexpr Double_t kPositionTolerance = 1E-6;
constexpr Double_t kMomentumTolerance = 1E-12;

////////////////////////////////////////////////////////////////////////////////
/// Generation mode constants
///
/// These correspond to the Shape parameter values:
/// - POINT (2): Fixed direction for all events
/// - RING (<0): Fixed polar angle, uniform azimuthal
/// - CIRCLE (0): Uniform within a cone
/// - STRIP (0<Ratio<=1): Rectangular angular region
///     Dphi = Ratio * TwoPi;
///     Dcos = Omega / Dphi;
////////////////////////////////////////////////////////////////////////////////
constexpr Double_t kModePoint  = 2.0;
constexpr Double_t kModeCircle = 0.0;

////////////////////////////////////////////////////////////////////////////////
/// \brief Check if shape parameter indicates point generation mode
/// \param[in] shape Shape parameter value
/// \return kTRUE if point mode
////////////////////////////////////////////////////////////////////////////////
inline Bool_t IsPoint(Double_t shape)
{
   return shape == kModePoint;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Check if shape parameter indicates circle generation mode
/// \param[in] shape Shape parameter value
/// \return kTRUE if circle mode
////////////////////////////////////////////////////////////////////////////////
inline Bool_t IsCircle(Double_t shape)
{
   return shape == kModeCircle;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Check if shape parameter indicates strip generation mode
/// \param[in] shape Shape parameter value
/// \return kTRUE if strip mode
////////////////////////////////////////////////////////////////////////////////
inline Bool_t IsStrip(Double_t shape)
{
   return shape > 0.0 && shape <= 1.0;
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Check if shape parameter indicates ring generation mode
/// \param[in] shape Shape parameter value
/// \return kTRUE if ring mode
////////////////////////////////////////////////////////////////////////////////
inline Bool_t IsRing(Double_t shape)
{
   return shape < 0.0;
}

////////////////////////////////////////////////////////////////////////////////
/// \struct Node_t
/// \brief Binary tree node for tracking multiple kinematic solutions
///
/// When a constrained particle can reach a detector via two different
/// momenta, both solutions are stored in a binary tree structure.
////////////////////////////////////////////////////////////////////////////////
struct Node_t {
   ROOT::Math::PxPyPzMVector fPV;      ///< Virtual system 4-momentum
   ROOT::Math::PxPyPzMVector fPDet;    ///< Detected particle 4-momentum
   Double_t                  fWeight;  ///< Weight (Jacobian contribution)
   Node_t                   *fLeft;    ///< Left child (second solution)
   Node_t                   *fRight;   ///< Right child (first solution)
   Node_t                   *fParent;  ///< Parent node

   /// Constructor
   Node_t(const ROOT::Math::PxPyPzMVector &p1,
          const ROOT::Math::PxPyPzMVector &p2,
          Double_t weight, Node_t *parent);

   /// Prevent copying to avoid ownership issues
   Node_t(const Node_t &) = delete;
   Node_t &operator=(const Node_t &) = delete;
};

////////////////////////////////////////////////////////////////////////////////
// Tree Management Functions
////////////////////////////////////////////////////////////////////////////////

/// \brief Recursively delete a tree and free all memory
/// \param[in] node Root of the tree to delete
void DeleteTree(Node_t *node);

/// \brief Create the first (root) node of the tree
Node_t *CreateFirst(Node_t *node,
                    const ROOT::Math::PxPyPzMVector &p1,
                    const ROOT::Math::PxPyPzMVector &p2,
                    Double_t weight);

/// \brief Create a right child node
Node_t *CreateRight(Node_t *node, Node_t *tmp,
                    const ROOT::Math::PxPyPzMVector &p1,
                    const ROOT::Math::PxPyPzMVector &p2,
                    Double_t weight);

/// \brief Create a left child node
Node_t *CreateLeft(Node_t *node, Node_t *tmp,
                   const ROOT::Math::PxPyPzMVector &p1,
                   const ROOT::Math::PxPyPzMVector &p2,
                   Double_t weight);

/// \brief Collect all root-to-leaf paths for 4-momenta
void CollectPaths(Int_t nBody, Node_t *node,
                  std::vector<ROOT::Math::PxPyPzMVector> &path,
                  std::vector<std::vector<ROOT::Math::PxPyPzMVector>> &paths);

/// \brief Collect all root-to-leaf paths for weights
void CollectPathsWeights(Int_t nBody, Node_t *node,
                         std::vector<Double_t> &path,
                         std::vector<std::vector<Double_t>> &paths);

////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////

/// \brief Phase space kinematic function F(x,y) = sqrt((1-x-y)^2 - 4xy)
/// \param[in] x First mass ratio squared (m1^2/M^2)
/// \param[in] y Second mass ratio squared (m2^2/M^2)
/// \return Kinematic function value
Double_t CalcKMFactor(Double_t x, Double_t y);

////////////////////////////////////////////////////////////////////////////////
// Core Physics Functions
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
void TwoBody(const ROOT::Math::PxPyPzMVector &S,
             Double_t m1, Double_t m2,
             ROOT::Math::PxPyPzMVector &p1,
             ROOT::Math::PxPyPzMVector &p2,
             TRandom3 *rng);

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
Bool_t TGenPointSpace(const ROOT::Math::PxPyPzMVector &S1,
                      Double_t m1, Double_t m2,
                      ROOT::Math::XYZVector vDet,
                      Int_t &solutions,
                      Double_t *jackPDF,
                      ROOT::Math::PxPyPzMVector *pDet,
                      ROOT::Math::PxPyPzMVector *pD2);

/// \brief Generate random direction vector within specified solid angle
///
/// \param[in] Omega Solid angle size [steradians]
/// \param[in] Ratio Shape parameter determining generation mode
/// \param[in] Vcenter Central direction vector
/// \param[out] vPoint Generated direction vector
/// \param[in] rng Pointer to TRandom3 random number generator (thread-safe)
void TGenVec(Double_t Omega, Double_t Ratio,
             ROOT::Math::XYZVector Vcenter,
             ROOT::Math::XYZVector &vPoint,
             TRandom3 *rng);

////////////////////////////////////////////////////////////////////////////////
// Main Generator Function
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
Int_t GenFANG(Int_t nBody,
              const ROOT::Math::PxPyPzMVector &S,
              const Double_t *masses,
              const Double_t *Om,
              const Double_t *Ratio,
              std::vector<ROOT::Math::XYZVector> V3Det,
              std::vector<std::vector<ROOT::Math::PxPyPzMVector>> &VecVecP,
              std::vector<Double_t> &vecWi,
              TRandom3 *rng);

} // namespace FANG

////////////////////////////////////////////////////////////////////////////////
/// \class TFANG
/// \brief User interface class for the FANG phase space generator
///
/// TFANG wraps the FANG namespace functions to provide an object-oriented
/// interface for phase space event generation with optional angular constraints.
///
/// Example usage (unconstrained):
/// ~~~{.cpp}
/// // Create generator for 3-body decay
/// TFANG gen;
/// Double_t masses[3] = {0.139, 0.139, 0.139};  // pion masses
/// ROOT::Math::PxPyPzMVector S(0, 0, 0, 1.0);   // parent at rest
/// gen.SetDecay(S, 3, masses);
///
/// // Generate single event
/// gen.Generate();
/// auto p0 = gen.GetDecay(0);  // 4-momentum of first particle
///
/// // Calculate phase space integral
/// Double_t ps, err;
/// gen.GetPhaseSpace(1000000, ps, err);
/// ~~~
///
/// Example usage (constrained):
/// ~~~{.cpp}
/// TFANG gen;
/// Double_t masses[3] = {0.139, 0.139, 0.139};
/// ROOT::Math::PxPyPzMVector S(0, 0, 0, 1.0);
/// gen.SetDecay(S, 3, masses);
///
/// // Add detector constraint
/// ROOT::Math::XYZVector detDir(0, 0, 1);
/// Double_t omega = 0.01;  // solid angle
/// Double_t ratio = 0.0;   // circle mode
/// gen.AddConstraint(detDir, omega, ratio);
///
/// // Generate and loop over solutions
/// if (gen.Generate() > 0) {
///    for (Int_t i = 0; i < gen.GetNSolutions(); ++i) {
///       Double_t w = gen.GetWeight(i);
///       auto p0 = gen.GetDecay(i, 0);
///    }
/// }
/// ~~~
////////////////////////////////////////////////////////////////////////////////
class TFANG {

private:
   Int_t                                  fNBody;      ///< Number of outgoing particles
   ROOT::Math::PxPyPzMVector              fS;          ///< Total 4-momentum of system
   std::vector<Double_t>                  fMasses;     ///< Particle masses
   std::vector<Double_t>                  fOmega;      ///< Solid angles for constraints
   std::vector<Double_t>                  fRatio;      ///< Shape parameters for constraints
   std::vector<ROOT::Math::XYZVector>     fV3Det;      ///< Direction vectors for constraints

   std::vector<std::vector<ROOT::Math::PxPyPzMVector>> fVecVecP;  ///< Output 4-momenta for all solutions
   std::vector<Double_t>                  fVecWi;      ///< Weights for each solution

   TRandom3                              *fRng;        ///< Random number generator
   Bool_t                                 fOwnRng;     ///< Whether we own the RNG

public:
   /// Default constructor
   TFANG();

   /// Constructor with external random number generator
   /// \param[in] rng Pointer to external TRandom3 (ownership not transferred)
   explicit TFANG(TRandom3 *rng);

   /// Destructor
   ~TFANG();

   /// Prevent copying
   TFANG(const TFANG &) = delete;
   TFANG &operator=(const TFANG &) = delete;

   ////////////////////////////////////////////////////////////////////////////
   // Configuration methods
   ////////////////////////////////////////////////////////////////////////////

   /// \brief Set decay configuration
   /// \param[in] S Total 4-momentum of the decaying system
   /// \param[in] nBody Number of outgoing particles
   /// \param[in] masses Array of particle masses, length nBody
   /// \return kTRUE if configuration is valid
   Bool_t SetDecay(const ROOT::Math::PxPyPzMVector &S, Int_t nBody, const Double_t *masses);

   /// \brief Add angular constraint for a particle
   /// \param[in] direction Direction vector for the constraint
   /// \param[in] omega Solid angle in steradians
   /// \param[in] ratio Shape parameter (2=point, 0=circle, 0<r<=1=strip, <0=ring)
   void AddConstraint(const ROOT::Math::XYZVector &direction, Double_t omega, Double_t ratio);

   /// \brief Clear all constraints (revert to unconstrained generation)
   void ClearConstraints();

   /// \brief Set random number generator seed
   /// \param[in] seed Seed value
   void SetSeed(UInt_t seed);

   ////////////////////////////////////////////////////////////////////////////
   // Event generation methods
   ////////////////////////////////////////////////////////////////////////////

   /// \brief Generate a single phase space event
   /// \return Number of solutions (0 if generation failed)
   Int_t Generate();

   ////////////////////////////////////////////////////////////////////////////
   // Accessors for generated event data
   ////////////////////////////////////////////////////////////////////////////

   /// \brief Get number of solutions from last generation
   /// \return Number of kinematic solutions
   Int_t GetNSolutions() const { return static_cast<Int_t>(fVecVecP.size()); }

   /// \brief Get weight for a specific solution (unconstrained mode)
   /// \return Weight for the single solution
   Double_t GetWeight() const;

   /// \brief Get weight for a specific solution (constrained mode)
   /// \param[in] iSolution Solution index
   /// \return Weight for the specified solution
   Double_t GetWeight(Int_t iSolution) const;

   /// \brief Get 4-momentum of particle (unconstrained mode, single solution)
   /// \param[in] iParticle Particle index (0 to nBody-1)
   /// \return 4-momentum of the particle
   ROOT::Math::PxPyPzMVector GetDecay(Int_t iParticle) const;

   /// \brief Get 4-momentum of particle for specific solution
   /// \param[in] iSolution Solution index
   /// \param[in] iParticle Particle index (0 to nBody-1)
   /// \return 4-momentum of the particle
   ROOT::Math::PxPyPzMVector GetDecay(Int_t iSolution, Int_t iParticle) const;

   /// \brief Get all 4-momenta for a solution
   /// \param[in] iSolution Solution index
   /// \return Vector of 4-momenta
   const std::vector<ROOT::Math::PxPyPzMVector> &GetDecays(Int_t iSolution) const;

   ////////////////////////////////////////////////////////////////////////////
   // Phase space integration methods
   ////////////////////////////////////////////////////////////////////////////

   /// \brief Calculate full (unconstrained) phase space integral with uncertainty
   ///
   /// Runs Monte Carlo integration to estimate the full phase space volume.
   /// This method ignores any constraints and calculates the Lorentz-invariant
   /// phase space for the decay.
   ///
   /// \param[in] nEvents Number of events for Monte Carlo integration (default 1E6)
   /// \param[out] phaseSpace Estimated phase space value
   /// \param[out] error Statistical uncertainty on phase space
   /// \return kTRUE if calculation succeeded
   Bool_t GetPhaseSpace(Long64_t nEvents, Double_t &phaseSpace, Double_t &error) const;

   /// \brief Calculate full phase space with default 1E6 events
   /// \param[out] phaseSpace Estimated phase space value
   /// \param[out] error Statistical uncertainty on phase space
   /// \return kTRUE if calculation succeeded
   Bool_t GetPhaseSpace(Double_t &phaseSpace, Double_t &error) const;

   /// \brief Calculate partial (constrained) phase space integral with uncertainty
   ///
   /// Runs Monte Carlo integration to estimate the partial phase space volume
   /// when angular constraints are applied. The result is the phase space
   /// restricted to the detector solid angles, multiplied by the product of
   /// all omega values (total solid angle factor).
   ///
   /// Requires constraints to be set via AddConstraint() before calling.
   ///
   /// \param[in] nEvents Number of events for Monte Carlo integration (default 1E6)
   /// \param[out] phaseSpace Estimated partial phase space value
   /// \param[out] error Statistical uncertainty on phase space
   /// \return kTRUE if calculation succeeded, kFALSE if no constraints set
   Bool_t GetPartialPhaseSpace(Long64_t nEvents, Double_t &phaseSpace, Double_t &error) const;

   /// \brief Calculate partial phase space with default 1E6 events
   /// \param[out] phaseSpace Estimated partial phase space value
   /// \param[out] error Statistical uncertainty on phase space
   /// \return kTRUE if calculation succeeded, kFALSE if no constraints set
   Bool_t GetPartialPhaseSpace(Double_t &phaseSpace, Double_t &error) const;

   ////////////////////////////////////////////////////////////////////////////
   // Utility methods
   ////////////////////////////////////////////////////////////////////////////

   /// \brief Check if generator is in constrained mode
   /// \return kTRUE if constraints are set
   Bool_t IsConstrained() const { return !fV3Det.empty(); }

   /// \brief Get number of constraints
   /// \return Number of detector constraints
   Int_t GetNConstraints() const { return static_cast<Int_t>(fV3Det.size()); }

   /// \brief Get number of particles in decay
   /// \return Number of outgoing particles
   Int_t GetNBody() const { return fNBody; }
};

#endif // ROOT_TFANG
