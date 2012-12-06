#ifndef ClingWorkAroundMissingDynamicScope
{
gROOT->ProcessLine(".L constructor.C+");
#else
#include "constructor.C"
void runConstructor()
{
#endif
TemplateArgument c;
A a;
a.doit(c);
A a2(c);
}
