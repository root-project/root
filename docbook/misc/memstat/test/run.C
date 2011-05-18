void run(bool only_compile = false)
{
   gSystem->Load("libMemStat");
   // calling a "leaker"
   if(!only_compile)
      gROOT->ProcessLine(".x leak_test.C++g");
}
