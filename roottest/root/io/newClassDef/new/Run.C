void Run() {

   gSystem->Load("libIoNewClassNewInheritMulti");
   gSystem->Load("libIoNewClassNewnamespace");
   gSystem->Load("libIoNewClassNewtemplate");
   gSystem->Load("libIoNewClassNewnstemplate");

#ifdef ClingWorkAroundMissingDynamicScope
#ifdef ClingWorkAroundFunctionForwardDeclarations
   gROOT->ProcessLine("#include \"functionsFwdDeclarations.h\"");
#endif
   gROOT->ProcessLine(
   "namespace_driver();"
   "template_driver();"
   "nstemplate_driver();"

   "if (! InheritMulti_driver() ) exit(1);"
                      );
#else
#ifdef ClingWorkAroundFunctionForwardDeclarations
   #include "functionsFwdDeclarations.h"
#endif
   namespace_driver();
   template_driver();
   nstemplate_driver();

   if (! InheritMulti_driver() ) exit(1);
#endif
}
