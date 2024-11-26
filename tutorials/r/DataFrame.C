/// \file
/// \ingroup tutorial_r
/// \notebook -nodraw
///
/// \macro_code
///
/// \author

void DataFrame()
{
   using namespace ROOT::R;
   // Creating variables
   TVectorD v1(3);
   std::vector<Double_t> v2 {0.101, 0.202, 0.303};
   std::array<Int_t,3>  v3{ {1,2,3} };
   std::list<std::string> names {"v1", "v2", "v3"};

   // Assigning values
   v1[0]=1;
   v1[1]=2;
   v1[2]=3;

   auto &r = TRInterface::Instance();

   // Creating dataframe object with its labels

   TRDataFrame  df1(Label["var1"]=v1,Label["var2"]=v2,Label["var3"]=v3,Label["strings"]=names);

   // Passing dataframe to R's environment

   r["df1"]<<df1;
   r<<"print(df1)";

   // Adding colunms to dataframe

   TVectorD v4(3);
   //filling the vector fro R's environment
   r["c(-1,-2,-3)"]>>v4;
   //adding new colunm to df1 with name var4
   df1["var4"]=v4;
   //updating df1 in R's environment
   r["df1"]<<df1;
   //printing df1
   r<<"print(df1)";

   // Getting dataframe from R's environment

   TRDataFrame df2;

   r<<"df2<-data.frame(v1=c(0.1,0.2,0.3),v2=c(3,2,1))";
   r["df2"]>>df2;

   TVectorD v(3);
   df2["v1"]>>v;
   v.Print();

   df2["v2"]>>v;
   v.Print();

   // Working with colunms between dataframes

   df2["v3"]<<df1["strings"];

   //updating df2 in R's environment
   r["df2"]<<df2;
   r<<"print(df2)";

   // Passing values from colunm v3 of df2 to var1 of df1
   df2["v3"]>>df1["var1"];

   // Updating df1 in R's environment
   r["df1"]<<df1;
   r<<"print(df1)";
}
