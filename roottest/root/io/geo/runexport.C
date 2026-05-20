void runexport(const char *geom,const char *comp="")
{
   gROOT->ProcessLine(Form(".x %s.C%s",geom,comp));
   gGeoManager->Export(Form("%s.export.root",geom));
}
