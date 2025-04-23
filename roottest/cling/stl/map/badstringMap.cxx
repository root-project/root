#ifndef ClingWorkAroundMissingDynamicScope
{
#include <map>
#include <string>

TClass *cl = gROOT->GetClass("map<string,double>");
if (cl->GetClassInfo()==0) {
   gROOT->ProcessLine(".L stringMapLoad.cxx+");
}

gROOT->ProcessLine(".L MyClass.C+");
// gSystem->Load("libLibrary.so");

#else
#include <map>
#include <string>

TClass *cl = gROOT->GetClass("map<string,double>");
long loader1 = (cl->GetClassInfo()==0) 
   ? gROOT->ProcessLine(".L stringMapLoad.cxx+")
   : 0;
#include "MyClass.C"

int badstringMap()
{
#endif

MyClass m;



map<string,double> &mm(m.Param());
mm["hello"]=3.14149;
cout << mm["hello"] << endl;

m.Param()["hello"]=3.14159;
cout<<m.Param()["hello"];
cout<<endl;

#ifndef ClingWorkAroundMissingDynamicScope
printf("(int) 0\n");
#endif
return 0;
}
