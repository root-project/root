// Author: Omar Zapata
#include<TRInterface.h>
#include<TRDataFrame.h>
#include<vector>
#include<array>

using namespace ROOT::R;

void DataFrame() {
//creating variables
    TVectorD v(3);
    std::vector<Double_t> sv(3);
//std::array<Int_t,3>  a{ {1,2,3} };

//assinging values
    v[0]=1;
    v[1]=2;
    v[2]=3;

    sv[0]=0.101;
    sv[1]=0.202;
    sv[2]=0.303;


    TRInterface &r=TRInterface::Instance();
// r.SetVerbose(kTRUE);
    std::list<std::string> names;
    names.push_back("v1");
    names.push_back("v2");
    names.push_back("v3");

    TRDataFrame  df(Label["var1"]=v,Label["var2"]=sv,Label["var3"]=sv,Label["strings"]=names);
    TRDataFrame  df2;


    r["df"]<<df;

//printting results
    std::cout<<"-----------Printing Results---------\n";
    r<<"print(df)";

    std::cout<<"------------------------------------\n";
    df["var1"]>>sv;

    r["v"]<<sv;
    r<<"print(v)";
    df["var3"]<<sv;

    df["var4"]<<sv;
    df["var5"]=names;
    r["df"]<<df;
    std::cout<<"------------------------------------\n";
    r<<"print(df)";

    //tests of operators between dataframe and r interface object
    std::cout<<"------------------------------------\n";
    r<<"df2<-data.frame(v1=c(1,2,3),v2=c('a','b','c'),v3=c(3,2,1))";
    r["df2"]>>df2;
    r["v2"]<<df2["v2"];
    r<<"print(v2)";
    r["v3"]<<df2["v3"];
    r<<"print(v3)";

    //tests between dataframes operator   
    std::cout<<"------------------------------------\n";
    df2["v4"]<<df2["v3"];
    df2["v2"]<<df2["v3"];
    r["df2"]<<df2;
    r<<"print(df2)";
    
    //the next line is not working
    df2["v5"]=df2["v1"];
    df2["v1"]=df2["v3"];
    df2["v6"]<<df2["v5"];
    r["df2"]<<df2;
    r<<"print(df2)";
    
     // the next line donk work, the operator >> is not supported between Bindings(FIXED NOW))
     df2["v6"]>>df2["v1"];
     
     //basic methods
     std::cout<<"------------------------------------\n";
     std::cout<<"nrows = "<<df2.GetNrows()<<std::endl;
     std::cout<<"ncols = "<<df2.GetNcols()<<std::endl;
     
     
     r["v5"]<<df2["v5"];
     r<<"print(v5)";
     df2["v5"]>>df["var1"];
     r["v5"]<<df2["v5"];
     r<<"print(v5)";
    
     
     //Error Handling
/*     std::cout<<"------------------------------------\n";
     try{
      r["qwe"]<<df2["qwe"];
     }   
     catch(Rcpp::index_out_of_bounds& __ex__){
        ::Error("operator=", "%s",__ex__.what());
        forward_exception_to_r( __ex__ ) ;
     }
     catch(...){::Error("operator=", "Can not assign in v5");}
*/     
}