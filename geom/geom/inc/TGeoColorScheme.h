//==============================================================================
// TGeoColorScheme.h
//==============================================================================

#ifndef ROOT_TGeoColorScheme
#define ROOT_TGeoColorScheme

#include "Rtypes.h"
#include <functional>

class TGeoVolume;
class TGeoMaterial;

enum class EGeoColorSet {
   kNatural = 0,
   kFlashy,
   kHighContrast
};

/**
 * TGeoColorScheme
 *
 * Strategy object used by TGeoManager::DefaultColors(const TGeoColorScheme*)
 * to assign visualization colors / transparency to volumes, typically after
 * GDML import where no colors are stored.
 *
 * This class is intended for runtime use only (not persistified). Users can
 * extend or override the default behavior via hooks (std::function) or by
 * subclassing and overriding virtual methods.
 */
class TGeoColorScheme {
public:
   using ColorHook_t = std::function<Int_t(const TGeoVolume *)>;
   using TranspHook_t = std::function<Int_t(const TGeoVolume *)>;
   using ZFallbackHook_t = std::function<Int_t(Int_t /*Z*/, EGeoColorSet /*set*/)>;

   explicit TGeoColorScheme(EGeoColorSet set = EGeoColorSet::kNatural);
   virtual ~TGeoColorScheme();

   /// Return >=0 to apply a ROOT color index (TColor index), <0 for "no decision".
   virtual Int_t Color(const TGeoVolume *vol) const;

   /// Return [0..100] to apply transparency, <0 to leave unchanged.
   virtual Int_t Transparency(const TGeoVolume *vol) const;

   /// Z-binned fallback. Users may override via SetZFallbackHook() or subclassing.
   virtual Int_t ColorForZ(Int_t Z, EGeoColorSet set) const;

   /// Hooks setters. Set to nullptr to disable.
   void SetColorHook(ColorHook_t h) { fColorHook = std::move(h); }
   void SetTransparencyHook(TranspHook_t h) { fTranspHook = std::move(h); }
   void SetZFallbackHook(ZFallbackHook_t h) { fZFallbackHook = std::move(h); }

   EGeoColorSet GetSet() const { return fSet; }
   void SetSet(EGeoColorSet s) { fSet = s; }

   /// Helper for user hooks: get material from volume with pointer checks.
   static const TGeoMaterial *GetMaterial(const TGeoVolume *vol);

private:
   EGeoColorSet fSet;

   ColorHook_t fColorHook = nullptr;
   TranspHook_t fTranspHook = nullptr;
   ZFallbackHook_t fZFallbackHook = nullptr;
};

#endif // ROOT_TGeoColorScheme
