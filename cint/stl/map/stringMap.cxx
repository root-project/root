#ifndef ClingWorkAroundUnnamedInclude
{
#include <map>
#include <string>
#else
#include <map>
#include <string>
void stringMap() {
#endif

TClass *cl = gROOT->GetClass("map<string,double>");
if (cl->GetClassInfo()==0) {
   gROOT->ProcessLine(".L stringMapLoad.cxx+");
}

map<string,double> m;
m["howdy"]=3.14159;
cout<<m["howdy"]<<endl;
#ifndef ClingWorkAroundUnnamedInclude
return 0;
#endif
}
