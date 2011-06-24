{
   gROOT->ProcessLine(".L templateName.cpp+");
   plot_my_i();
   gSystem->Sleep(2000);
   makeclass();
}
