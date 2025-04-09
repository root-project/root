{
// Fill out the code of the actual test
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L sample_bx_classes.C+");
#endif
   gROOT->ProcessLine(".x sample_reader.C");
}
