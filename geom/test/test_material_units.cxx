#include <gtest/gtest.h>

#include <TGeoManager.h>
#include <TGeoElement.h>
#include <TGeoMaterial.h>
#include <TGeoSystemOfUnits.h>
#include <TGeant4SystemOfUnits.h>
#include <TGeant4PhysicalConstants.h>
#include <cstring>
#include <sstream>
#include <iostream>
#include <iomanip>

struct PDG_material {
   std::string name;
   double Z;           // atomic number
   double radlen;      // [cm]
   double intlen;      // [cm]
   double mass;        // g/mole
   double density;     // g/cm3
   double length_unit; // For pdg, tgeo: 1, g4: 10
   PDG_material() = default;
   PDG_material(PDG_material &&) = default;
   PDG_material(const PDG_material &) = default;
   ~PDG_material() = default;
   PDG_material &operator=(const PDG_material &) = default;
};

// Radiation/interaction length must be within 2 % of pdg values
// The formula used by TGeo and G4 is not full exact compared to the measured PDG values
// Values are in units of %!
double tolerance_density = 0.01;
double tolerance_radlen = 0.3;
double tolerance_intlen = 4.0;
bool debug = false;

int check_mat(const PDG_material &pdg, const PDG_material &mat, bool have_tolerance, bool verbose = false)
{
   double length_norm = mat.length_unit;
   const char *units = fabs(length_norm - 10.0) < 1e-10 ? "G4" : "TGeo";
   const char *length = fabs(length_norm - 10.0) < 1e-10 ? " [mm] " : " [cm] ";
   double dens = mat.density;
   double radlen = mat.radlen / length_norm;
   double intlen = mat.intlen / length_norm;
   std::stringstream log;
   int num_err = 0;

   double diff_dens = fabs(dens / pdg.density - 1.0);
   if (diff_dens > (have_tolerance ? tolerance_density / 100e0 : 1e-15)) {
      log << "TEST FAILED " << std::setw(16) << std::left << mat.name << " Units: " << units
          << " Density out of tolerance " << diff_dens << " PDG: " << pdg.density << " TGeo:" << dens << std::endl;
      ++num_err;
   }
   double diff_radlen = fabs(radlen / pdg.radlen - 1.0);
   if (diff_radlen > (have_tolerance ? tolerance_radlen / 100e0 : 1e-15)) {
      log << "TEST FAILED " << std::setw(16) << std::left << mat.name << " Units: " << units
          << " RadLen out of tolerance " << diff_radlen << " PDG: " << pdg.radlen << " TGeo:" << radlen << std::endl;
      ++num_err;
   }
   double diff_intlen = fabs(intlen / pdg.intlen - 1.0);
   if (diff_intlen > (have_tolerance ? tolerance_intlen / 100e0 : 1e-15)) {
      log << "TEST FAILED " << std::setw(16) << std::left << mat.name << " Units: " << units
          << " Intlen out of tolerance " << diff_intlen << " PDG: " << pdg.intlen << " TGeo:" << intlen << std::endl;
      ++num_err;
   }
   if (verbose) {
      log << ((0 == num_err) ? "TEST PASSED " : "TEST FAILED ");
      log << std::setw(14) << std::left << pdg.name << " vs. " << std::setw(14) << std::left << mat.name
          << " Units: " << std::setw(4) << std::left << units << " Deviation "
          << " density: " << std::setw(2) << std::setprecision(2) << diff_dens * 100.0 << " % "
          << " RadLen:  " << std::setw(8) << std::setprecision(2) << diff_radlen * 100.0 << " % "
          << " IntLen:  " << std::setw(4) << std::setprecision(2) << diff_intlen * 100.0 << " % " << std::endl;
      if (num_err > 0 || debug) {
         log << mat.name << std::endl
             << "\t\t Density:  \t\t" << dens << " [g/cm^3]" << std::endl
             << "\t\t Radiation   Length: \t" << mat.radlen << length << std::endl
             << "\t\t Interaction Length: \t" << mat.intlen << length << std::endl
             << std::endl;
      }
      std::cout << log.str();
   }
   EXPECT_EQ(num_err, 0);
   return num_err;
}

void build_material(const PDG_material &pdg, PDG_material result[], bool verbose = false)
{
   double length_unit = (TGeoManager::GetDefaultUnits() == TGeoManager::kG4Units) ? TGeant4Unit::cm : TGeoUnit::cm;
   std::string nam = pdg.name + ((TGeoManager::GetDefaultUnits() == TGeoManager::kG4Units) ? "_G4" : "_TGeo");
   TGeoElementTable *table = gGeoManager->GetElementTable();
   TGeoElement *elt = table->GetElement(pdg.Z);

   if (verbose)
      elt->Print();
   TGeoMaterial *mat = new TGeoMaterial(("Mat_" + nam).c_str(), elt, pdg.density); // Material from element
   if (verbose)
      mat->Print();
   TGeoMixture *mix1 = new TGeoMixture(("Mix_1_" + nam).c_str(), 1, pdg.density); // Mixture from element
   mix1->AddElement(elt, 1.0);
   mix1->ComputeDerivedQuantities();
   TGeoMixture *mix2 = new TGeoMixture(("Mix_2_" + nam).c_str(), 1, pdg.density); // Mixture from material
   mix2->AddElement(mat, 1.0);
   TGeoMixture *mix3 = new TGeoMixture(("Mix_3_" + nam).c_str(), 1, pdg.density); // Mixture from mixture
   mix3->AddElement(mix1, 1.0);
   result[0] = {{mat->GetName()}, mat->GetZ(),       mat->GetRadLen(), mat->GetIntLen(),
                mat->GetA(),      mat->GetDensity(), length_unit};
   result[1] = {{mix1->GetName()}, mix1->GetZ(),       mix1->GetRadLen(), mix1->GetIntLen(),
                mix1->GetA(),      mix1->GetDensity(), length_unit};
   result[2] = {{mix2->GetName()}, mix2->GetZ(),       mix2->GetRadLen(), mix2->GetIntLen(),
                mix2->GetA(),      mix2->GetDensity(), length_unit};
   result[3] = {{mix3->GetName()}, mix3->GetZ(),       mix3->GetRadLen(), mix3->GetIntLen(),
                mix3->GetA(),      mix3->GetDensity(), length_unit};
}

/**
   Small test to verify that radiation length and nuclear interaction length
   are computed correctly when ROOT uses G4 units.
   \author Makus Frank
*/
TEST(Geometry, MaterialUnits)
{
   // Values from the PDG website: https://pdg.lbl.gov/2020/AtomicNuclearProperties
   static constexpr size_t num_mat = 3;
   bool verbose = false;
   PDG_material pdg[num_mat] = {{{"Si"}, 14, 9.370, 46.52, 28.085, 2.329, TGeoUnit::cm},
                                {{"Fe"}, 26, 1.757, 16.77, 55.854, 7.874, TGeoUnit::cm},
                                {{"U"}, 92, 0.3166, 11.03, 238.0289, 18.95, TGeoUnit::cm}};
   PDG_material tgeo_mat[num_mat][4];
   PDG_material g4_mat[num_mat][4];

   TGeoManager::LockDefaultUnits(kFALSE);
   TGeoManager::SetDefaultUnits(TGeoManager::kRootUnits);
   TGeoManager::LockDefaultUnits(kTRUE);
   // Delete the current TGeoManager (if any) and create the new one only after units were changed
   if (gGeoManager)
      delete gGeoManager;
   gGeoManager = new TGeoManager();
   std::cout << " Using ROOT system of units. " << std::endl;

   for (size_t i = 0; i < num_mat; ++i)
      build_material(pdg[i], tgeo_mat[i], verbose | debug);

   TGeoManager::LockDefaultUnits(kFALSE);
   TGeoManager::SetDefaultUnits(TGeoManager::kG4Units);
   TGeoManager::LockDefaultUnits(kTRUE);
   // Delete the current TGeoManager (if any) and create the new one only after units were changed
   if (gGeoManager)
      delete gGeoManager;
   gGeoManager = new TGeoManager();
   std::cout << " Using Geant4 system of units. " << std::endl;

   for (size_t i = 0; i < num_mat; ++i)
      build_material(pdg[i], g4_mat[i], verbose | debug);

   int nerrs = 0;
   for (size_t i = 0; i < num_mat; ++i) {
      // Test pdg against materials in TGeo units
      for (size_t j = 0; j < 4; ++j)
         nerrs += check_mat(pdg[i], tgeo_mat[i][j], true, verbose | debug);
      // Test pdg against materials in G4 units
      for (size_t j = 0; j < 4; ++j)
         nerrs += check_mat(pdg[i], g4_mat[i][j], true, verbose | debug);
      // Test materials in G4 units against materials in TGeo units
      for (size_t j = 0; j < 4; ++j)
         nerrs += check_mat(tgeo_mat[i][j], g4_mat[i][j], true, verbose | debug);
   }
   if (verbose | debug)
      std::cout << std::endl
                << ((0 == nerrs) ? "TEST PASSED " : "TEST FAILED ") << nerrs << " failures detected." << std::endl;
   EXPECT_EQ(nerrs, 0);
}
