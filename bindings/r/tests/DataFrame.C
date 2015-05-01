// Author: Omar Zapata
#include<TRInterface.h>
#include<TRDataFrame.h>
#include<vector>
#include<array>

using namespace ROOT::R;

void DataFrame(){
//creating variables
TVectorD v(3);
std::vector<Double_t> sv(3);
std::array<Int_t,3>  a{ {1,2,3} };

//assinging values
v[0]=0.01;
v[1]=1.01;
v[2]=2.01;

sv[0]=0.101;
sv[1]=0.202;
sv[2]=0.303;


TRInterface &r=TRInterface::Instance();
// r.SetVerbose(kTRUE);
std::list<std::string> names;
names.push_back("v1");
names.push_back("v2");
names.push_back("v3");
//TRDataFrame  df(Label["var1"]=v);
TRDataFrame  df(Label["var"]=v,Label["var2"]=sv,Label["var3"]=sv,Label["strings"]=names);

r["df"]<<df;
//printting results
std::cout<<"-----------Printing Results---------\n";
r<<"print(df)";

}
