void export(const char *geom) {
   gROOT->ProcessLine(Form(".x %s.C",geom));
   gGeoManager->Export(Form("%s.root",geom));
}