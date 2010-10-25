{
gROOT->ProcessLine(".L constructor.C+");
TemplateArgument c;
A a;
a.doit(c);
A a2(c);
}
