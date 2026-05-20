//==============================================================================
// TGeoColorScheme.cxx
//==============================================================================

/**
 * @class TGeoColorScheme
 * @ingroup Geometry_classes
 * @brief Strategy class for assigning colors/transparency to geometry volumes.
 *
 * @details
 * Helper "strategy" class used by TGeoManager::DefaultColors() to assign
 * visualization attributes (line color and optional transparency) to imported
 * geometries (e.g. GDML), where the source format does not encode colors.
 *
 * @par Design goals
 * @li Backward compatibility: calling TGeoManager::DefaultColors() with no
 *     arguments behaves like today (default "natural" scheme).
 * @li Reasonable defaults for detector geometries: first try name-based material
 *     classification (e.g. copper, aluminium, steel, FR4/G10, kapton/polyimide,
 *     gases), then fall back to a Z-binned lookup.
 * @li User extensibility without requiring subclassing: users can provide hooks
 *     (std::function) that override the default decisions partially or fully.
 *     Hooks are runtime-only (not persistified).
 *
 * @par How TGeoManager::DefaultColors() uses a scheme
 * For each volume:
 * @li scheme->Color(vol) is queried:
 *     @li return >= 0 : use returned ROOT color index (TColor index)
 *     @li return <  0 : caller may ignore or use its own fallback
 * @li scheme->Transparency(vol) is queried:
 *     @li return in [0..100] : set volume transparency
 *     @li return < 0         : do not change transparency
 *
 * @par Default behavior
 * @li Color(vol):
 *     @li If a user color hook is installed, it is called first. If it returns
 *         >= 0, that color is used.
 *     @li Otherwise, try name-based overrides (tokens like "copper", "steel",
 *         "fr4", "g10", "kapton", "argon", "c6f14", ...).
 *     @li If still unresolved, fall back to ColorForZ(effectiveZ). Users can
 *         override the Z fallback via SetZFallbackHook() or by subclassing and
 *         overriding ColorForZ().
 * @li Transparency(vol):
 *     @li If a user transparency hook is installed and returns >= 0, use it.
 *     @li Otherwise, the default makes very low-density materials (gases)
 *         semi-transparent (60), preserving historical behavior.
 *
 * @par User extensibility examples
 * Example A: Tweak just copper-like materials (leave everything else default)
 * @code{.cpp}
 * TGeoColorScheme cs(EGeoColorSet::kNatural);
 * cs.SetColorHook([](const TGeoVolume* v) -> Int_t {
 *   const TGeoMaterial* m = TGeoColorScheme::GetMaterial(v);
 *   if (!m || !m->GetName()) return -1;
 *   std::string n = m->GetName();
 *   std::transform(n.begin(), n.end(), n.begin(), ::tolower);
 *   if (n.find("copper") != std::string::npos || n.find("_cu") != std::string::npos)
 *     return TColor::GetColor(0.90f, 0.55f, 0.30f); // custom copper tint
 *   return -1; // let defaults handle the rest
 * });
 * gGeoManager->DefaultColors(&cs);
 * @endcode
 *
 * Example B: Override Z-fallback mapping (keep name overrides)
 * @code{.cpp}
 * TGeoColorScheme cs(EGeoColorSet::kNatural);
 * cs.SetZFallbackHook([](Int_t Z, EGeoColorSet) -> Int_t {
 *   float t = std::min(1.f, std::max(0.f, Z / 100.f));
 *   return TColor::GetColor(t, t, t); // grayscale by Z
 * });
 * gGeoManager->DefaultColors(&cs);
 * @endcode
 *
 * Example C: Full custom policy via subclassing (optional)
 * @code{.cpp}
 * class MyScheme : public TGeoColorScheme {
 * public:
 *   using TGeoColorScheme::TGeoColorScheme;
 *   Int_t ColorForZ(Int_t Z, EGeoColorSet set) const override { ... }
 *   Int_t Color(const TGeoVolume* vol) const override { ... }
 * };
 * MyScheme cs(EGeoColorSet::kFlashy);
 * gGeoManager->DefaultColors(&cs);
 * @endcode
 *
 * @par Notes
 * @li This class is intended for runtime use; it is not persistified because
 *     hooks are user-provided callables which are not I/O friendly.
 * @li The default name token rules are heuristic and aimed at typical HEP
 *     detector material naming conventions; users can refine them locally via
 *     hooks or subclassing.
 *
 * @see TGeoManager::DefaultColors
 */

#include <string>
#include <algorithm>

#include "TGeoColorScheme.h"
#include "TColor.h"
#include "TGeoVolume.h"
#include "TGeoMedium.h"
#include "TGeoMaterial.h"

//---- local helpers -----------------------------------------------------------

namespace {

inline Int_t RGB(int r, int g, int b)
{
   return TColor::GetColor(r / 255.f, g / 255.f, b / 255.f);
}

inline Int_t ClampTransp(Int_t t)
{
   if (t < 0)
      return -1;
   if (t > 100)
      return 100;
   return t;
}

inline std::string Norm(std::string s)
{
   std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      if (c == '-' || c == ' ' || c == '.' || c == '/' || c == '\\' || c == ',')
         return '_';
      return (char)std::tolower(c);
   });
   return s;
}

inline bool Has(const std::string &s, const char *token)
{
   return s.find(token) != std::string::npos;
}

enum class EMatClass {
   kGas,
   kAir,
   kVacuum,
   kCopper,
   kAluminium,
   kTungsten,
   kSteel,
   kIron,
   kFR4_G10,
   kCarbonFiber,
   kEpoxy,
   kPE,
   kBoratedPE,
   kPVC,
   kPolyurethane,
   kPolyimide,
   kNomex,
   kTyvek,
   kFoam,
   kRohacell,
   kSiO2,
   kAlN,
   kCeramic,
   kCables,
   kThermalScreen,
   kUnknown
};

EMatClass ClassFromName(const TGeoMaterial *mat)
{
   if (!mat || !mat->GetName())
      return EMatClass::kUnknown;
   const std::string n = Norm(mat->GetName());

   // gases/fluids
   if (Has(n, "vacuum"))
      return EMatClass::kVacuum;
   if (Has(n, "air"))
      return EMatClass::kAir;
   if (Has(n, "c6f14") || Has(n, "cf4") || Has(n, "co2") || Has(n, "argon") || Has(n, "gas") || Has(n, "dead_"))
      return EMatClass::kGas;

   // metals
   if (Has(n, "hmp_w") || Has(n, "tungsten") || Has(n, "_w_") || Has(n, "_w"))
      return EMatClass::kTungsten;
   if (Has(n, "hmp_cu") || Has(n, "copper") || Has(n, "_cu"))
      return EMatClass::kCopper;
   if (Has(n, "hmp_al") || Has(n, "aluminium") || Has(n, "aluminum") || Has(n, "2219") || Has(n, "_al"))
      return EMatClass::kAluminium;

   if (Has(n, "steel") || Has(n, "stainless"))
      return EMatClass::kSteel;
   if (Has(n, "cast_iron") || (Has(n, "iron") && !Has(n, "iridium")))
      return EMatClass::kIron;

   // PCB / laminates
   if (Has(n, "g10") || Has(n, "fr4") || Has(n, "textolit"))
      return EMatClass::kFR4_G10;

   // composites / organics
   if (Has(n, "carbon_fibre") || Has(n, "carbon_fiber"))
      return EMatClass::kCarbonFiber;
   if (Has(n, "epoxy"))
      return EMatClass::kEpoxy;

   if (Has(n, "borated") && (Has(n, "polyethyl") || Has(n, "polyethylene") || Has(n, "pe")))
      return EMatClass::kBoratedPE;

   if (Has(n, "polyimide") || Has(n, "kapton"))
      return EMatClass::kPolyimide;
   if (Has(n, "polyurethane"))
      return EMatClass::kPolyurethane;
   if (Has(n, "pvc"))
      return EMatClass::kPVC;
   if (Has(n, "polyethyl") || Has(n, "polyethylene"))
      return EMatClass::kPE;
   if (Has(n, "nomex"))
      return EMatClass::kNomex;
   if (Has(n, "tyvek"))
      return EMatClass::kTyvek;
   if (Has(n, "foam"))
      return EMatClass::kFoam;
   if (Has(n, "roha") || Has(n, "rohacell"))
      return EMatClass::kRohacell;

   // ceramics / oxides / nitrides
   if (Has(n, "sio2") || Has(n, "quartz") || Has(n, "silica"))
      return EMatClass::kSiO2;
   if (Has(n, "aluminium_nitride") || Has(n, "aln"))
      return EMatClass::kAlN;
   if (Has(n, "ceramic"))
      return EMatClass::kCeramic;

   // non-dominant services
   if (Has(n, "cables") || Has(n, "cable"))
      return EMatClass::kCables;
   if (Has(n, "thermalscreen") || Has(n, "thermal_screen") || Has(n, "thermal"))
      return EMatClass::kThermalScreen;

   return EMatClass::kUnknown;
}

Int_t ClassColor(EMatClass c, EGeoColorSet set)
{
   switch (set) {
   case EGeoColorSet::kNatural:
      switch (c) {
      case EMatClass::kGas: return RGB(190, 230, 240);
      case EMatClass::kAir: return RGB(210, 210, 210);
      case EMatClass::kVacuum: return RGB(230, 230, 230);

      case EMatClass::kCopper: return RGB(184, 115, 51);
      case EMatClass::kAluminium: return RGB(190, 190, 195);
      case EMatClass::kTungsten: return RGB(45, 45, 50);
      case EMatClass::kSteel: return RGB(120, 125, 135);
      case EMatClass::kIron: return RGB(90, 95, 105);

      case EMatClass::kFR4_G10: return RGB(35, 115, 55);
      case EMatClass::kCarbonFiber: return RGB(35, 35, 38);
      case EMatClass::kEpoxy: return RGB(140, 95, 55);

      case EMatClass::kPE: return RGB(220, 215, 205);
      case EMatClass::kBoratedPE: return RGB(215, 205, 185);
      case EMatClass::kPVC: return RGB(200, 200, 205);
      case EMatClass::kPolyurethane: return RGB(230, 225, 185);
      case EMatClass::kPolyimide: return RGB(185, 140, 50);
      case EMatClass::kNomex: return RGB(220, 185, 90);
      case EMatClass::kTyvek: return RGB(235, 240, 245);
      case EMatClass::kFoam: return RGB(245, 245, 240);
      case EMatClass::kRohacell: return RGB(245, 240, 235);

      case EMatClass::kSiO2: return RGB(230, 235, 240);
      case EMatClass::kAlN: return RGB(235, 235, 225);
      case EMatClass::kCeramic: return RGB(225, 225, 220);

      case EMatClass::kCables: return RGB(55, 55, 60);
      case EMatClass::kThermalScreen: return RGB(195, 205, 215);

      default: return -1;
      }

   case EGeoColorSet::kFlashy:
      switch (c) {
      case EMatClass::kGas: return RGB(120, 220, 255);
      case EMatClass::kAir: return RGB(170, 240, 255);
      case EMatClass::kVacuum: return RGB(245, 245, 245);

      case EMatClass::kCopper: return RGB(255, 140, 60);
      case EMatClass::kAluminium: return RGB(210, 210, 220);
      case EMatClass::kTungsten: return RGB(80, 80, 90);
      case EMatClass::kSteel: return RGB(160, 160, 175);
      case EMatClass::kIron: return RGB(130, 135, 150);

      case EMatClass::kFR4_G10: return RGB(0, 200, 70);
      case EMatClass::kCarbonFiber: return RGB(120, 120, 130);
      case EMatClass::kEpoxy: return RGB(255, 180, 80);

      case EMatClass::kBoratedPE: return RGB(255, 240, 160);
      case EMatClass::kPolyimide: return RGB(255, 200, 40);
      case EMatClass::kTyvek: return RGB(230, 250, 255);
      case EMatClass::kFoam: return RGB(255, 230, 230);
      case EMatClass::kRohacell: return RGB(230, 255, 230);

      case EMatClass::kSiO2: return RGB(210, 235, 255);
      case EMatClass::kAlN: return RGB(255, 255, 210);

      case EMatClass::kCables: return RGB(255, 80, 80);
      case EMatClass::kThermalScreen: return RGB(180, 220, 255);

      default: return -1;
      }

   case EGeoColorSet::kHighContrast:
      switch (c) {
      case EMatClass::kGas: return RGB(20, 90, 110);
      case EMatClass::kAir: return RGB(70, 70, 75);
      case EMatClass::kVacuum: return RGB(110, 110, 115);

      case EMatClass::kCopper: return RGB(110, 45, 10);
      case EMatClass::kAluminium: return RGB(90, 95, 105);
      case EMatClass::kTungsten: return RGB(30, 30, 35);
      case EMatClass::kSteel: return RGB(60, 65, 75);
      case EMatClass::kIron: return RGB(50, 55, 65);

      case EMatClass::kFR4_G10: return RGB(10, 80, 30);
      case EMatClass::kCarbonFiber: return RGB(25, 25, 28);
      case EMatClass::kEpoxy: return RGB(85, 55, 25);

      case EMatClass::kBoratedPE: return RGB(110, 95, 55);
      case EMatClass::kPolyimide: return RGB(100, 70, 10);
      case EMatClass::kTyvek: return RGB(80, 95, 110);
      case EMatClass::kFoam: return RGB(100, 80, 80);
      case EMatClass::kRohacell: return RGB(85, 105, 85);

      case EMatClass::kSiO2: return RGB(70, 85, 100);
      case EMatClass::kAlN: return RGB(90, 90, 70);

      case EMatClass::kCables: return RGB(80, 20, 20);
      case EMatClass::kThermalScreen: return RGB(70, 85, 100);

      default: return -1;
      }
   }

   return -1;
}

Int_t NameOverrideColor(const TGeoMaterial *mat, EGeoColorSet set)
{
   const auto cls = ClassFromName(mat);
   return ClassColor(cls, set);
}

} // namespace

//==============================================================================
// TGeoColorScheme implementation
//==============================================================================

TGeoColorScheme::TGeoColorScheme(EGeoColorSet set) : fSet(set) {}

TGeoColorScheme::~TGeoColorScheme() = default;

const TGeoMaterial *TGeoColorScheme::GetMaterial(const TGeoVolume *vol)
{
   if (!vol)
      return nullptr;
   const TGeoMedium *med = vol->GetMedium();
   if (!med)
      return nullptr;
   return med->GetMaterial();
}

Int_t TGeoColorScheme::Color(const TGeoVolume *vol) const
{
   if (!vol)
      return -1;

   // User override hook first
   if (fColorHook) {
      const Int_t c = fColorHook(vol);
      if (c >= 0)
         return c;
   }

   const TGeoMaterial *mat = GetMaterial(vol);
   if (!mat)
      return -1;

   // Built-in name token override
   const Int_t cName = NameOverrideColor(mat, fSet);
   if (cName >= 0)
      return cName;

   // Built-in Z fallback
   const Int_t Z = (Int_t)mat->GetZ();
   return ColorForZ(Z, fSet);
}

Int_t TGeoColorScheme::Transparency(const TGeoVolume *vol) const
{
   if (!vol)
      return -1;

   if (fTranspHook) {
      const Int_t t = fTranspHook(vol);
      if (t >= 0)
         return ClampTransp(t);
   }

   const TGeoMaterial *mat = GetMaterial(vol);
   if (!mat)
      return -1;

   // Historical default: gases transparent by density.
   if (mat->GetDensity() < 0.1)
      return 60;

   return -1;
}

Int_t TGeoColorScheme::ColorForZ(Int_t Z, EGeoColorSet set) const
{
   if (fZFallbackHook) {
      const Int_t c = fZFallbackHook(Z, set);
      if (c >= 0)
         return c;
   }

   if (Z <= 0)
      return kGray + 1;
   if (Z > 109)
      Z = 109;

   switch (set) {
   case EGeoColorSet::kNatural:
      if (Z <= 2)
         return kGray + 2;
      if (Z <= 6)
         return RGB(60, 60, 60);
      if (Z <= 8)
         return RGB(90, 110, 130);
      if (Z <= 10)
         return RGB(120, 140, 155);
      if (Z <= 14)
         return RGB(40, 85, 170);
      if (Z <= 16)
         return RGB(140, 160, 120);
      if (Z <= 20)
         return RGB(175, 175, 175);
      if (Z <= 26)
         return RGB(110, 120, 130);
      if (Z <= 28)
         return RGB(95, 105, 115);
      if (Z == 29)
         return RGB(184, 115, 51);
      if (Z <= 30)
         return RGB(150, 150, 155);
      if (Z <= 35)
         return RGB(160, 130, 70);
      if (Z <= 40)
         return RGB(140, 145, 150);
      if (Z <= 50)
         return RGB(120, 125, 130);
      if (Z <= 56)
         return RGB(105, 110, 120);
      if (Z <= 74)
         return RGB(45, 45, 50);
      if (Z <= 79)
         return RGB(130, 110, 30);
      if (Z <= 82)
         return RGB(90, 80, 95);
      return RGB(80, 85, 90);

   case EGeoColorSet::kFlashy:
      if (Z <= 2)
         return RGB(120, 220, 255);
      if (Z <= 6)
         return RGB(255, 120, 220);
      if (Z <= 10)
         return RGB(80, 140, 255);
      if (Z <= 14)
         return RGB(80, 200, 255);
      if (Z <= 20)
         return RGB(80, 240, 140);
      if (Z <= 26)
         return RGB(255, 240, 100);
      if (Z <= 30)
         return RGB(255, 170, 60);
      if (Z <= 40)
         return RGB(255, 90, 90);
      if (Z <= 56)
         return RGB(140, 255, 200);
      if (Z <= 74)
         return RGB(200, 140, 255);
      return RGB(255, 160, 220);

   case EGeoColorSet::kHighContrast:
      if (Z <= 2)
         return RGB(30, 30, 30);
      if (Z <= 6)
         return RGB(70, 20, 20);
      if (Z <= 10)
         return RGB(15, 45, 90);
      if (Z <= 14)
         return RGB(10, 65, 120);
      if (Z <= 20)
         return RGB(25, 80, 35);
      if (Z <= 26)
         return RGB(90, 70, 15);
      if (Z <= 30)
         return RGB(110, 45, 10);
      if (Z <= 40)
         return RGB(110, 15, 35);
      if (Z <= 56)
         return RGB(55, 20, 85);
      if (Z <= 74)
         return RGB(35, 35, 45);
      return RGB(60, 50, 70);
   }

   return kGray;
}
