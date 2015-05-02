#include<TRInterface.h>
 
using namespace ROOT::R;
void DataFrame()
{
////////////////////////
//creating variables//
////////////////////////
TVectorD v1(3);
std::vector<Double_t> v2(3);
std::array<Int_t,3>  v3{ {1,2,3} };
std::list<std::string> names;
 
//////////////////////
//assigning values//
//////////////////////
v1[0]=1;
v1[1]=2;
v1[2]=3;
 
v2[0]=0.101;
v2[1]=0.202;
v2[2]=0.303;
 
names.push_back("v1");
names.push_back("v2");
names.push_back("v3");

TRInterface &r=TRInterface::Instance(); 

/////////////////////////////////////////////
//creating dataframe object with its labels//
/////////////////////////////////////////////
 
TRDataFrame  df1(Label["var1"]=v1,Label["var2"]=v2,Label["var3"]=v3,Label["strings"]=names);
 
//////////////////////////////////////////////
//Passing dataframe to R's environment//
//////////////////////////////////////////////
 
r["df1"]<<df1;
r<<"print(df1)";

////////////////////////////////
//Adding colunms to dataframe //
////////////////////////////////

TVectorD v4(3);
//filling the vector fro R's environment
r["c(-1,-2,-3)"]>>v4;
//adding new colunm to df1 with name var4
df1["var4"]=v4;
//updating df1 in R's environment
r["df1"]<<df1;
//printing df1
r<<"print(df1)";


//////////////////////////////////////////
//Getting dataframe from R's environment//
//////////////////////////////////////////
TRDataFrame df2;

r<<"df2<-data.frame(v1=c(0.1,0.2,0.3),v2=c(3,2,1))";
r["df2"]>>df2;

TVectorD v(3);
df2["v1"]>>v;
v.Print();

df2["v2"]>>v;
v.Print();



///////////////////////////////////////////
//Working with colunms between dataframes//
///////////////////////////////////////////

df2["v3"]<<df1["strings"];

//updating df2 in R's environment
r["df2"]<<df2;
r<<"print(df2)";

//passing values from colunm v3 of df2 to var1 of df1 
df2["v3"]>>df1["var1"];
//updating df1 in R's environment
r["df1"]<<df1;
r<<"print(df1)";

}