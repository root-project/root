/*************************************************************************
 * Copyright (C) 2013-2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRDataFrame.h>
//______________________________________________________________________________
/* Begin_Html
<center><h2>TRDataFrame class</h2></center>

DataFrame? is a very important datatype in R and in ROOTR we have a class to manipulate<br>
dataframes called TRDataFrame, with a lot of very useful operators overloaded to work with TRDataFrame's objects<br>
in a similar way that in the R environment but from c++ in ROOT.<br>
Example:<br>
<br>
Lets to create need data to play with dataframe features<br>
End_Html

Begin_Html
<hr>
///////////////////////////////////<br>
//creating variables///<br>
//////////////////////////////////<br>
End_Html
TVectorD v1(3);
std::vector<Double_t> v2(3);
std::array<Int_t,3>  v3{ {1,2,3} };
std::list<std::string> names;
 
Begin_Html
//////////////////////////////////<br>
//assigning values//<br>
//////////////////////////////////<br>
End_Html
v1[0]=1;
v1[1]=2;
v1[2]=3;
 
v2[0]=0.101;
v2[1]=0.202;
v2[2]=0.303;
 
names.push_back("v1");
names.push_back("v2");
names.push_back("v3");
 
ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
Begin_Html
<hr>
End_Html

Begin_Html
In R the dataframe have associate to every column a label, 
in ROOTR you can have the same label using the class ROOT::R::Label to create a TRDataFrame where you data
have a label associate.
End_Html

Begin_Html
<hr>
///////////////////////////////////////////////////////////////////<br>
//creating dataframe object with its labels//<br>
///////////////////////////////////////////////////////////////////<br>
End_Html
using namespace ROOT::R;
TRDataFrame  df1(Label["var1"]=v1,Label["var2"]=v2,Label["var3"]=v3,Label["strings"]=names);
Begin_Html
//////////////////////////////////////////////////////////////////<br>
//Passing dataframe to R's environment//<br>
/////////////////////////////////////////////////////////////////<br>
End_Html
r["df1"]<<df1;
r<<"print(df1)";
Begin_Html
<hr>
End_Html


Output
var1  var2 var3 strings
1    1 0.101    1      v1
2    2 0.202    2      v2
3    3 0.303    3      v3


Begin_Html
Manipulating data between dataframes
End_Html

Begin_Html
<hr>
/////////////////////////////////////////////////////<br>
//Adding colunms to dataframe //<br>
/////////////////////////////////////////////////////<br>
End_Html
 
TVectorD v4(3);
//filling the vector fro R's environment
r["c(-1,-2,-3)"]>>v4;
//adding new colunm to df1 with name var4
df1["var4"]=v4;
//updating df1 in R's environment
r["df1"]<<df1;
//printing df1
r<<"print(df1)";
Begin_Html
<hr>
End_Html


Output
var1  var2 var3 strings var4
1    1 0.101    1      v1   -1
2    2 0.202    2      v2   -2
3    3 0.303    3      v3   -3


Getting data frames from R's environment

Begin_Html
<hr>
/////////////////////////////////////////////////////////////////////<br>
//Getting dataframe from R's environment//<br>
/////////////////////////////////////////////////////////////////////<br>
End_Html
ROOT::R::TRDataFrame df2;
 
r<<"df2<-data.frame(v1=c(0.1,0.2,0.3),v2=c(3,2,1))";
r["df2"]>>df2;
 
TVectorD v(3);
df2["v1"]>>v;
v.Print();
 
df2["v2"]>>v;
v.Print();
Begin_Html
<hr>
End_Html

Output
Vector (3)  is as follows
 
     |        1  |
------------------
   0 |0.1 
   1 |0.2 
   2 |0.3 
 
Vector (3)  is as follows
 
     |        1  |
------------------
   0 |3 
   1 |2 
   2 |1


Begin_Html
<hr>
/////////////////////////////////////////////////////////////////////////<br>
//Working with colunms between dataframes//<br>
/////////////////////////////////////////////////////////////////////////<br>
End_Html
 
df2["v3"]<<df1["strings"];
 
//updating df2 in R's environment
r["df2"]<<df2;
r<<"print(df2)";
Begin_Html
<hr>
End_Html


Output
v1 v2 v3
1 0.1  3 v1
2 0.2  2 v2
3 0.3  1 v3


Begin_Html
<hr>
//////////////////////////////////////////////////////////////////////<br>
//Working with colunms between dataframes//<br>
//////////////////////////////////////////////////////////////////////<br>
End_Html
 
//passing values from colunm v3 of df2 to var1 of df1 
df2["v3"]>>df1["var1"];
//updating df1 in R's environment
r["df1"]<<df1;
r<<"print(df1)";
Begin_Html
<hr>
End_Html


Output
var1  var2 var3 strings var4
1   v1 0.101    1      v1   -1
2   v2 0.202    2      v2   -2
3   v3 0.303    3      v3   -3
*/


using namespace ROOT::R;
ClassImp(TRDataFrame)


//______________________________________________________________________________
TRDataFrame::TRDataFrame(): TObject()
{
    df = Rcpp::DataFrame::create();
}

//______________________________________________________________________________
TRDataFrame::TRDataFrame(const TRDataFrame &_df): TObject(_df)
{
    df = _df.df;
}

//______________________________________________________________________________
TRDataFrame::Binding TRDataFrame::operator[](const TString &name)
{
    return Binding(df,name);
}
