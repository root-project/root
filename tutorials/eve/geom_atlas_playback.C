/// \file
/// \ingroup tutorial_eve
/// Plays back event-recording of a root session running geom_atlas.C tutorial.
/// [ Recorded using "new TGRecorder" command. ]
///
/// Script:
/// - type: .x geom_atlas.C
/// - demonstrate rotation (left-mouse), zoom (right-mouse left-right)
/// - show wireframe (w), smooth (r, default) and outline (t) render modes
/// - show flip of background color dark-light-dark (e pressed twice)
/// - disable clipping plane in GL-viewer panel
/// - open "Scene" list-tree and further "Geometry scene"
/// - disable drawing of muon system and then calorimeters
/// - select tracker geometry top-node and increase drawing depth
/// - re-enable clipping plane and zoom into pixel detector.
///
/// \macro_code
///
/// \author Matevz Tadel

void geom_atlas_playback()
{
   auto r = new TRecorder("http://mtadel.home.cern.ch/mtadel/geom_atlas_recording.root");
}
