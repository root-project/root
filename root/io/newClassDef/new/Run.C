void Run() {

   gSystem->Load("./namespace");
   gSystem->Load("./template");
   gSystem->Load("./nstemplate");
   gSystem->Load("./InheritMulti");

#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
   "namespace_driver();"
   "template_driver();"
   "nstemplate_driver();"

   "if (! InheritMulti_driver() ) exit(1);"
                      );
#else
   namespace_driver();
   template_driver();
   nstemplate_driver();
   
   if (! InheritMulti_driver() ) exit(1);
#endif   
}
