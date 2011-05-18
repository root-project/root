void na49geomfile() {
//   Before executing this macro, the file makegeometry.C must have been executed
//
   gBenchmark->Start("geometry");
   TFile na("na49.root","RECREATE");
   TGeometry *n49 =(TGeometry*)gROOT->FindObject("na49");
   n49->Write();
   na.Write();
   gBenchmark->Show("geometry");
}
