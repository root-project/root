/**
 * \file geomColors.C
 * \ingroup tutorial_geom
 * Script demonstrating geometry color schemes.
 *
 *
 * \macro_code
 *
 * \author andrei.gheata@cern.ch
 */

#include <string>
#include <algorithm>

#include "TROOT.h"
#include "TColor.h"
#include "TControlBar.h"
#include "TGeoManager.h"
#include "TGeoColorScheme.h"

EGeoColorSet gLastSet = EGeoColorSet::kNatural;
Bool_t gTransparent = kFALSE;

void geomAlice_itsv()
{
   TGeoManager::Import("http://root.cern/files/alice2.root");
   gGeoManager->SetVisLevel(4);
   gGeoManager->GetVolume("ITSV")->Draw("ogl");
}

void help()
{
   printf("In the viewer window:\n"
          " - de-select \"Reset on update\"\n"
          " - in the \"Clipping\" tab, select \"Plane\"\n"
          " - rotate the image as uou wish and zoom using the mouse wheel\n"
          " - click on the different default color schemes in control bar menu\n"
          " - the \"gray\" function demonstrates overriding the Z-based fallback colors\n"
          " - the \"override\" function demonstrates color override from a scheme by using a hook\n"
          " - the \"transparency\" function demonstrates transparency override\n");
}

//______________________________________________________________________________
void natural()
{
   // Predefined "natural" color scheme
   if (!gGeoManager)
      return;
   gLastSet = EGeoColorSet::kNatural;
   TGeoColorScheme cs(gLastSet);
   gGeoManager->DefaultColors(&cs);
}

//______________________________________________________________________________
void flashy()
{
   // Predefined "flashy" color scheme
   if (!gGeoManager)
      return;
   gLastSet = EGeoColorSet::kFlashy;
   TGeoColorScheme cs(gLastSet);
   gGeoManager->DefaultColors(&cs);
}

//______________________________________________________________________________
void high_contrast()
{
   // Predefined "high-contrast" color scheme
   if (!gGeoManager)
      return;
   gLastSet = EGeoColorSet::kHighContrast;
   TGeoColorScheme cs(gLastSet);
   gGeoManager->DefaultColors(&cs);
}

//______________________________________________________________________________
void gray()
{
   // Gray palette override of the Z-binned fallback
   if (!gGeoManager)
      return;
   TGeoColorScheme cs(gLastSet);
   cs.SetZFallbackHook([](Int_t Z, EGeoColorSet) -> Int_t {
      float g = std::min(1.f, Z / 100.f);
      return TColor::GetColor(g, g, g);
   });
   gGeoManager->DefaultColors(&cs);
}

//______________________________________________________________________________
void transparent()
{
   // Transparency override for a color scheme
   if (!gGeoManager)
      return;
   gTransparent = !gTransparent;
   TGeoColorScheme cs(gLastSet);
   if (gTransparent) {
      cs.SetTransparencyHook([](const TGeoVolume *v) -> Int_t {
         const TGeoMaterial *m = TGeoColorScheme::GetMaterial(v);
         if (!m)
            return -1;

         // Base glass-like transparency for "everything"
         Int_t tr = 85; // 0=opaque, 100=fully transparent

         // Make typical gases/fluids even more "invisible"
         if (m->GetDensity() < 0.1)
            tr = 95;

         // Optional: slightly reduce transparency for cables/services so they remain visible
         if (m->GetName()) {
            std::string n = m->GetName();
            std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return (char)std::tolower(c); });

            if (n.find("cable") != std::string::npos || n.find("cables") != std::string::npos)
               tr = 70;

            // Optional: keep heavy metals a bit less transparent to preserve structure
            if (n.find("tungsten") != std::string::npos || n.find("_w") != std::string::npos)
               tr = 75;
         }

         return tr;
      });
   } else {
      cs.SetTransparencyHook([](const TGeoVolume *v) -> Int_t { return 0; });
   }

   gGeoManager->DefaultColors(&cs);
}

//______________________________________________________________________________
void color_override()
{
   // Color override based on volume-related properties
   if (!gGeoManager)
      return;
   TGeoColorScheme cs(gLastSet);
   cs.SetColorHook([](const TGeoVolume *v) -> Int_t {
      const TGeoMaterial *m = TGeoColorScheme::GetMaterial(v);
      if (!m || !m->GetName())
         return -1;
      if (std::string(m->GetName()).find("ITS_GEN") != std::string::npos)
         return TColor::GetColor(0.9f, 0.55f, 0.3f);
      return -1; // fallback to defaults
   });
   gGeoManager->DefaultColors(&cs);
}

//______________________________________________________________________________
void geomColors()
{
   // root[0] .x geomColors.C
   //
   // This opens a GL viewer showing the detector with the user-defined colors.
   // In the viewer window:
   // - de-select "Reset on update"
   // - in the "Clipping" tab, select "Plane"
   // - rotate the image as uou wish and zoom using the mouse wheel
   // - click on the different default color schemes (natural, flashy, high-contrast) in control bar menu
   // - the "Gray fallback" function demonstrates overriding the Z-based fallback colors
   // - the "Color override" function demonstrates color override from a scheme by using a hook
   // - the "Transparency" function demonstrates transparency override from a scheme using a hook.
   //   Transparency is attached to TGeoMaterial, so it affects all volumes sharing the same material

   TControlBar *bar = new TControlBar("vertical", "Geometry color schemes", 10, 10);
   bar->AddButton("How to run  ", "help()", "Instructions for running this macro");
   bar->AddButton("Natural        ", "natural()",
                  "A natural color scheme with name-based/Z-binned lookup for material classification");
   bar->AddButton("Flashy         ", "flashy()", "Flashy, high-contrast, presentation-friendly colors");
   bar->AddButton("High contrast  ", "high_contrast()", "Darker, saturated colors suited for light backgrounds");
   bar->AddButton("Gray fallback  ", "gray()", "Demonstrates overriding the Z-based fallback colors");
   bar->AddButton("Color override  ", "color_override()", "Demonstrates color override from a scheme by using a hook");
   bar->AddButton("Transparency ON/OFF  ", "transparent()",
                  "Demonstrates transparency override from a scheme by using a hook");
   bar->Show();
   gROOT->SaveContext();
   geomAlice_itsv();
}
