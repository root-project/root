/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we demonstrate RVec helpers for physics computations such
/// as angle differences \f$\Delta \phi\f$, the distance in the \f$\eta\f$-\f$\phi\f$
/// plane \f$\Delta R\f$ and the invariant mass.
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

   auto phi1 = Take(phis, idx[0]);
   auto phi2 = Take(phis, idx[1]);
   auto dphi = DeltaPhi(phi1, phi2);

   std::cout << "DeltaPhi(phi1 = " << phi1 << ",\n"
             << "         phi2 = " << phi2 << ")\n"
             << " = " << dphi << "\n";

   // The DeltaR helper is similar to the DeltaPhi helper and computes the distance
   // in the \f$\eta\f$-\f$\phi\f$ plane.
   RVec<float> etas = {2.4, -1.5, 1.0, 0.0};

   auto eta1 = Take(etas, idx[0]);
   auto eta2 = Take(etas, idx[1]);
   auto dr = DeltaR(eta1, eta2, phi1, phi2);

   std::cout << "\nDeltaR(eta1 = " << eta1 << ",\n"
             << "       eta2 = " << eta2 << ",\n"
             << "       phi1 = " << phi1 << ",\n"
             << "       phi2 = " << phi2 << ")\n"
             << " = " << dr << "\n";

   // The InvariantMasses helper computes the invariant mass of a two particle system
   // given the properties transverse momentum (pt), rapidity (eta), azimuth (phi)
   // and mass.
   RVec<float> pt3 = {40, 20, 30};
   RVec<float> eta3 = {2.5, 0.5, -1.0};
   RVec<float> phi3 = {-0.5, 0.0, 1.0};
   RVec<float> mass3 = {10, 10, 10};

   RVec<float> pt4 = {20, 10, 40};
   RVec<float> eta4 = {0.5, -0.5, 1.0};
   RVec<float> phi4 = {0.0, 1.0, -1.0};
   RVec<float> mass4 = {2, 2, 2};

   auto invMass = InvariantMasses(pt3, eta3, phi3, mass3, pt4, eta4, phi4, mass4);

   std::cout << "\nInvariantMass(pt1 = " << pt3 << ",\n"
             << "              eta1 = " << eta3 << ",\n"
             << "              phi1 = " << phi3 << ",\n"
             << "              mass1 = " << mass3 << ",\n"
             << "              pt2 = " << pt4 << ",\n"
             << "              eta2 = " << eta4 << ",\n"
             << "              phi2 = " << phi4 << ",\n"
             << "              mass2 = " << mass4 << ")\n"
             << " = " << invMass << "\n";

   // The InvariantMass helper also accepts a single set of (pt, eta, phi, mass) vectors. Then,
   // the invariant mass of all particles in the collection is computed.

   auto invMass2 = InvariantMass(pt3, eta3, phi3, mass3);

   std::cout << "\nInvariantMass(pt = " << pt3 << ",\n"
             << "              eta = " << eta3 << ",\n"
             << "              phi = " << phi3 << ",\n"
             << "              mass = " << mass3 << ")\n"
             << " = " << invMass2 << "\n";
}
