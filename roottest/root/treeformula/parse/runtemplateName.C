{
   gROOT->ProcessLine(".L templateName.cpp+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("plot_my_i();");
   gROOT->ProcessLine("gSystem->Sleep(2000);");
   gROOT->ProcessLine("makeclass();");
#else
   plot_my_i();
   gSystem->Sleep(2000);
   makeclass();
#endif
}
