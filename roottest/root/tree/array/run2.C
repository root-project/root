{
gROOT->ProcessLine(".L TestObj.cpp+");
gROOT->ProcessLine(".L save.C+");
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine("save();");
#else
save();
#endif
}
