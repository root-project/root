#ifdef ClingWorkAroundMissingDynamicScope
#include "MyOpClass.C"
int badOperator() 
{
#else
{
gROOT->ProcessLine(".L MyOpClass.C+");
#endif
MyOpClass obj;
obj.value()["33"];
const char *val = obj.value()["33"];
fprintf(stderr,"val==%s\n",val);
return strcmp(val,"33")!=0;
}
