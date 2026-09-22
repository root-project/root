{
gROOT->ProcessLine(".L t01.C+");
gROOT->ProcessLine(".L t02.C+");
t02 t;
TObject * const * op = t.getVal();
if (op && *op) {
   TObject o = **op;
   delete *op;
}
delete[] op;
}
