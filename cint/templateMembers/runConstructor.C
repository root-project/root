{
gROOT->ProcessLine(".L constructor.C+");
C c;
A a;
a.doit(c);
A a2(c);
}
