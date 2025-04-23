{
gROOT->ProcessLine(".O 0");
gROOT->ProcessLine(".L loopbreak.C");
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine("loopbreak();");
#else
loopbreak();
#endif

}
