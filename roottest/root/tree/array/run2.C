{
gROOT->ProcessLine(".L TestObj.cpp+");
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine("save();");
#else
save();
#endif
}
