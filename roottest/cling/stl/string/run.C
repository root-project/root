#ifndef ClingWorkAroundMissingDynamicScope
{
#else
#include "t01.C"
void run()
{
#endif
#ifndef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(".L t01.C+g");
#endif
#include <string>

string mystring("test string");

if (!t01(mystring)) {
   fprintf(stderr,"fail to pass by reference\n");
   gSystem->Exit(1);
}
if (!t01val(mystring) ) {
   fprintf(stderr,"fail to pass by value\n");
   gSystem->Exit(1);
}
if (!t01p(&mystring) ) {
   fprintf(stderr,"fail to pass by address\n");
   gSystem->Exit(1);
}

}
