void Run() {

   gSystem->Load("libIoNewClassInheritMulti");
   gSystem->Load("libIoNewClassnamespace");
   gSystem->Load("libIoNewClasstemplate");

#ifdef ClingWorkAroundMissingDynamicScope
#ifdef ClingWorkAroundFunctionForwardDeclarations
   gROOT->ProcessLine("#include \"driverFunctionsFwdDecl.h\"");
#endif
   gROOT->ProcessLine(
   "namespace_driver();"
   "template_driver();"
   //nstemplate_driver();
   "if (! InheritMulti_driver() ) exit(1);"
                      );
#else
#ifdef ClingWorkAroundFunctionForwardDeclarations
   #include "driverFunctionsFwdDecl.h"
#endif
   namespace_driver();
   template_driver();
   //nstemplate_driver();
   if (! InheritMulti_driver() ) exit(1);
#endif

   gROOT->ProcessLine(".L array.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("array_driver();");
#else
   array_driver();
#endif
}
