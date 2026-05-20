//==============================================================================
// TGeoColorScheme.h
//==============================================================================

#ifndef ROOT_TGeoColorScheme
#define ROOT_TGeoColorScheme

#include "Rtypes.h"
#include <functional>

class TGeoVolume;
class TGeoMaterial;

/**
 * @enum EGeoColorSet
 * @brief Enumeration of predefined geometry color schemes.
 *
 * These values select the built-in default coloring policy used by
 * TGeoColorScheme when assigning colors to geometry volumes.
 *
 * @see TGeoColorScheme
 */
enum class EGeoColorSet {
   kNatural = 0, ///< Natural, material-inspired colors (default)
   kFlashy,      ///< Bright, high-contrast colors for presentations
   kHighContrast ///< Dark, saturated colors for light backgrounds
};

/**
 * @class TGeoColorScheme
 * @brief Strategy object for assigning colors and transparency to geometry volumes.
 *
 * This class is used by TGeoManager::DefaultColors(const TGeoColorScheme*)
 * to assign visualization colors and transparency to geometry volumes,
 * typically after GDML import where no color information is stored.
 *
 * The default implementation combines:
 *  - name-based material classification (e.g. metals, polymers, gases),
 *  - a Z-binned fallback lookup when no name-based rule applies.
 *
 * This class is intended for runtime use only (not persistified).
 * Users can extend or override the default behavior by:
 *  - installing hooks via std::function, or
 *  - subclassing and overriding virtual methods.
 *
 * @see TGeoManager::DefaultColors
 */
class TGeoColorScheme {
public:
   /// Type of user hook for overriding color assignment.
   using ColorHook_t = std::function<Int_t(const TGeoVolume *)>;

   /// Type of user hook for overriding transparency assignment.
   using TranspHook_t = std::function<Int_t(const TGeoVolume *)>;

   /// Type of user hook for overriding the Z-based fallback coloring.
   using ZFallbackHook_t = std::function<Int_t(Int_t /*Z*/, EGeoColorSet /*set*/)>;

   /**
    * @brief Constructor.
    *
    * \param set  Initial color set selection (natural, flashy, high-contrast).
    */
   explicit TGeoColorScheme(EGeoColorSet set = EGeoColorSet::kNatural);

   /// @brief Virtual destructor.
   virtual ~TGeoColorScheme();

   /**
    * @brief Compute the color for a given volume.
    *
    * This method is called by TGeoManager::DefaultColors() for each volume.
    * The default implementation:
    *  - calls the user color hook if installed,
    *  - applies name-based material overrides,
    *  - falls back to ColorForZ() if no rule matches.
    *
    * @param vol  Geometry volume to be colored.
    * @return     ROOT color index (>=0) to apply, or <0 for "no decision".
    *
    * @see Transparency
    * @see ColorForZ
    * @see TGeoManager::DefaultColors
    */
   virtual Int_t Color(const TGeoVolume *vol) const;

   /**
    * @brief Compute the transparency for a given volume.
    *
    * The default implementation:
    *  - calls the user transparency hook if installed,
    *  - makes very low-density materials (e.g. gases) semi-transparent.
    *
    * @param vol  Geometry volume.
    * @return     Transparency value in [0..100] to apply, or <0 to leave unchanged.
    *
    * @see Color
    * @see TGeoManager::DefaultColors
    */
   virtual Int_t Transparency(const TGeoVolume *vol) const;

   /**
    * @brief Compute fallback color based on material effective Z.
    *
    * This method is used when no name-based material override applies.
    * Users may override this behavior via SetZFallbackHook() or by
    * subclassing and overriding this method.
    *
    * @param Z    Effective atomic number of the material.
    * @param set  Active color set selection.
    * @return     ROOT color index (>=0) to apply.
    *
    * @see SetZFallbackHook
    */
   virtual Int_t ColorForZ(Int_t Z, EGeoColorSet set) const;

   /**
    * @brief Set a user hook for color assignment.
    *
    * The hook is called before any built-in logic.
    * Returning a value <0 delegates the decision to the default implementation.
    *
    * @param h  Color hook (set to nullptr to disable).
    *
    * @see Color
    */
   void SetColorHook(ColorHook_t h) { fColorHook = std::move(h); }

   /**
    * @brief Set a user hook for transparency assignment.
    *
    * The hook is called before the default transparency logic.
    * Returning a value <0 delegates the decision to the default implementation.
    *
    * @param h  Transparency hook (set to nullptr to disable).
    *
    * @see Transparency
    */
   void SetTransparencyHook(TranspHook_t h) { fTranspHook = std::move(h); }

   /**
    * @brief Set a user hook for Z-based fallback coloring.
    *
    * The hook is called before the built-in Z-binned lookup.
    * Returning a value <0 delegates the decision to the default implementation.
    *
    * @param h  Z-fallback hook (set to nullptr to disable).
    *
    * @see ColorForZ
    */
   void SetZFallbackHook(ZFallbackHook_t h) { fZFallbackHook = std::move(h); }

   /// @brief Get the active color set.
   EGeoColorSet GetSet() const { return fSet; }

   /// @brief Set the active color set.
   void SetSet(EGeoColorSet s) { fSet = s; }

   /**
    * @brief Retrieve the material associated with a geometry volume.
    *
    * This helper performs all necessary pointer checks and may be safely
    * used inside user hooks.
    *
    * @param vol  Geometry volume.
    * @return     Pointer to the associated material, or nullptr if unavailable.
    */
   static const TGeoMaterial *GetMaterial(const TGeoVolume *vol);

private:
   EGeoColorSet fSet; ///< Active color set selection

   ColorHook_t fColorHook = nullptr;         ///< Optional user hook for color assignment
   TranspHook_t fTranspHook = nullptr;       ///< Optional user hook for transparency
   ZFallbackHook_t fZFallbackHook = nullptr; ///< Optional user hook for Z fallback
};

#endif // ROOT_TGeoColorScheme
