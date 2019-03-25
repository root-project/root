/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we demonstrate RVec helpers for physics computations such
/// as angle differences \f$\Delta \phi\f$ and invariant mass.
///
/// \macro_code
/// \macro_output
///
/// \date March 2019
/// \author Stefan Wunsch

using namespace ROOT::VecOps;

void vo007_PhysicsHelpers()
{
   // The DeltaPhi helper computes the closest angle between angles.
   // This means that the resulting value is in the range [-pi, pi].

   // Note that the helper also supports to compute the angle difference of an
   // RVec and a scalar and two scalars. In addition, the computation of the
   // difference and the behaviour at the boundary can be adjusted to radian and
   // degrees.
   RVec<float> phis = {0.0, 1.0, -0.5, M_PI + 1.0};
   auto idx = Combinations(phis, 2);

   auto phi_1 = Take(phis, idx[0]);
   auto phi_2 = Take(phis, idx[1]);
   auto dphi = DeltaPhi(phi_1, phi_2);

   std::cout << "Phi values: " << phis << std::endl;
   for(std::size_t i = 0; i < idx[0].size(); i++) {
      std::cout << "DeltaPhi(" << phis[idx[0][i]] << ", " << phis[idx[1][i]]
                << ") = " << dphi[i] << std::endl;
   }

   // The InvariantMass helper computes the invariant mass of a two particle system
   // given the properties transverse momentum (pt), rapidity (eta), azimuth (phi)
   // and mass.
   RVec<float> pt1 = {40, 20, 30};
   RVec<float> eta1 = {2.5, 0.5, -1.0};
   RVec<float> phi1 = {-0.5, 0.0, 1.0};
   RVec<float> mass1 = {10, 10, 10};

   RVec<float> pt2 = {20, 10, 40};
   RVec<float> eta2 = {0.5, -0.5, 1.0};
   RVec<float> phi2 = {0.0, 1.0, -1.0};
   RVec<float> mass2 = {2, 2, 2};

   auto invMass = InvariantMass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2);

   std::cout << std::endl;
   for(std::size_t i = 0; i < pt1.size(); i++) {
      std::cout << "InvariantMass("
                << pt1[i] << ", " << eta1[i] << ", " << phi1[i] << ", " << mass1[i] << ", "
                << pt2[i] << ", " << eta2[i] << ", " << phi2[i] << ", " << mass2[i]
                << ") = " << invMass[i] << std::endl;
   }

   // The helper also accepts a single set of (pt, eta, phi, mass) vectors. Then,
   // the invariant mass of all particles in the collection is computed.

   auto invMass2 = InvariantMass(pt1, eta1, phi1, mass1);

   std::cout << std::endl;
   std::cout << "InvariantMass(" << pt1 << ", " << eta1 << ", " << phi1 << ", "
             << mass1 << ") = " << invMass2 << std::endl;
}
