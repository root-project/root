int runtemplateMembers(bool compiled=false) {
   gROOT->ProcessLine(".L templateMembers.C+");
   if (compiled) gROOT->ProcessLine(".L templateMembersCode.C+"); 
   else gROOT->ProcessLine(".L templateMembersCode.C"); 
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("templateMembersCode();");
#else
   templateMembersCode();
#endif
   return 0;
}
