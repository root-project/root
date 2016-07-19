/// \file
/// \ingroup tutorial_geom
///  Before executing this macro, the file makegeometry.C must have been executed
///
/// \macro_code
///
/// \author Andrei Gheata

void na49geomfile() {
   gBenchmark->Start("geometry");
   TGeometry *n49 =(TGeometry*)gROOT->FindObject("na49");
   if (n49) {
      TFile na("na49.root","RECREATE");
      n49->Write();
      na.Write();
   }
   gBenchmark->Show("geometry");
}
