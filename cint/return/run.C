{
// Will fail
gROOT->ProcessLine(".L t01.C+");

gROOT->ProcessLine(".L t02.C+");
t02 t;
TObject o = **t.getVal();
TObject * const * op = t.getVal();
}
