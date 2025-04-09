{
gROOT->ProcessLine(".L twodimvar.C+");

#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
   "twrite(\"twodim_99.root\",99);"
   "twrite(\"twodim_0.root\",0);"
   "tread(\"twodim_0.root\");"
   "tread(\"twodim_99.root\");"
   "tscan(\"twodim_0.root\");"
   "tscan(\"twodim_99.root\");"
   );
#else
twrite("twodim_99.root",99);
twrite("twodim_0.root",0);

tread("twodim_0.root");

tread("twodim_99.root");

tscan("twodim_0.root");
tscan("twodim_99.root");
#endif
}
