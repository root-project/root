{
// Fill out the code of the actual test
gROOT->ProcessLine(".L ToplevelClass.C+");

Bool_t result;

gROOT->ProcessLine(".L Toplevel.C");

result = WriteToplevel();
result &= ReadToplevel();

gROOT->ProcessLine(".U Toplevel.C");
gROOT->ProcessLine(".L Toplevel.C+");

result &= WriteToplevel();
result &= ReadToplevel();

return !result; // invert value for Makefile purpose

}
