{
   gErrorIgnoreLevel = kError;

   #include "refClasses.cxx+"
   printf("Direct loading\n");
   #include "refFunction.C"
   refFunction();
   printf("ROOT loading\n");
   gROOT->ProcessLine(".L refFunction.C");
   refFunction();
   printf("ACLiC loading\n");
   gROOT->ProcessLine(".L refFunction.C+");
   refFunction();
   return 0;
}
