{
#include <map>
#include <string>

TClass *cl = gROOT->GetClass("map<string,double>");
if (cl->GetClassInfo()==0) {
   gROOT->ProcessLine(".L stringMapLoad.cxx+");
}

map<string,double> m;
m["howdy"]=3.14159;
cout<<m["howdy"]<<endl;
return 0;
}
