/// \file
/// \ingroup tutorial_geom
/// Exports a geometry in step format. Root must be compiled using geocad option ON.
///
/// The macro just calls the main conversion method of the TGeoToStep interface.
///
/// \author Andrei Gheata

void geo2stp()
{
   TString tutdir = gROOT->GetTutorialDir();
   TGeoManager::SetVerboseLevel(0);
   gROOT->ProcessLine(".x " + tutdir + "/geom/rootgeom.C(false)");
   // Create the TGeoToStep interface
   TGeoToStep geo2step(gGeoManager);
   // Write the geometry to a step file
   geo2step.CreateGeometry("rootgeom.stp");
}
