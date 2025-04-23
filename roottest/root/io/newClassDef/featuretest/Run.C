{

gROOT->ProcessLine(".L typeidtest.C++");
#ifdef ClingWorkAroundMissingDynamicScope
if (gROOT->ProcessLine("typeidtest();")==0) gApplication->Terminate(1);
#else
if (typeidtest()==0) gApplication->Terminate(1);
#endif

gROOT->ProcessLine(".x Class.C++");
gROOT->ProcessLine(".x ClassTrick.C++");

return 0;
}
