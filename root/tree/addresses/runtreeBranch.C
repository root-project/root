{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L userClass.C+");
   gROOT->ProcessLine(".x treeBranch.C");
   gROOT->ProcessLine(".x treeBranch.C+");
}
