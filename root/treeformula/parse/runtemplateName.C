{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L templateName.cpp+");
   plot_my_i();
   gSystem->Sleep(1000);
   makeclass();
}
