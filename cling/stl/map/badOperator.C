#ifndef ClingWorkAroundMissingDynamicScope
{
gROOT->ProcessLine(".L MyOpClass.C+");
#else
#include "MyOpClass.C"
void badOperator() 
{
#endif
MyOpClass obj;
obj.value()["33"];
const char *val = obj.value()["33"];
fprintf(stderr,"val==%s\n",val);
#ifndef ClingWorkAroundMissingDynamicScope
return strcmp(val,"33")!=0;
#else
 if ( strcmp(val,"33")!=0 ) {
    fprintf(stderr,"Error val is %s which is not 33\n",val);
 }
#endif
}
