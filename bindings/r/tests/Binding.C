// Author: Omar Zapata
#include<TRInterface.h>
#include<vector>
#include<array>

Double_t Function(Double_t x)
{
  return x/(x-1);
}

Double_t Fun(Double_t x)
{
  return x/(x-1);
}
//overloaded function to test the operator <<
Int_t Fun(Int_t x)
{
  return x-1;
}

void Binding(){
//creating variables
TVectorD v(3);
std::vector<Double_t> sv(3);
std::array<Int_t,3>  a{ {1,2,3} };
TString str("ROOTR");
TMatrixD m(2,2);
Int_t i=9;
Double_t d=2.013;
Float_t f=0.013;

//assinging values
v[0]=0.01;
v[1]=1.01;
v[2]=2.01;

sv[0]=0.101;
sv[1]=0.202;
sv[2]=0.303;

m[0][0]=0.01;
m[0][1]=1.01;
m[1][0]=2.01;
m[1][1]=3.01;

ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
// r.SetVerbose(kTRUE);

//testing operators binding
r["a"]<<1;
r["v"]<<v;
r["sv"]<<sv;
r["m"]<<m;
r["b"]<<123.456;
r["i"]<<i;
r["d"]<<d;
r["f"]<<f;
r["array"]<<a;
r["s"]<<"ROOT";


//printting results
std::cout<<"-----------Printing Results---------\n";
r<<"print(a)";
std::cout<<"--------------------\n";
r<<"print(v)";
std::cout<<"--------------------\n";
r<<"print(sv)";
std::cout<<"--------------------\n";
r<<"print(m)";
std::cout<<"--------------------\n";
r<<"print(b)";
std::cout<<"--------------------\n";
r<<"print(i)";
std::cout<<"--------------------\n";
r<<"print(d)";
std::cout<<"--------------------\n";
r<<"print(f)";
std::cout<<"--------------------\n";
r<<"print(s)";
std::cout<<"--------------------\n";
r<<"print(array)";
std::cout<<"--------------------\n";

//reassigning the variable s
r["s"]<<str;//string with string
r<<"print(s)";

// std::cout<<"--------------------\n";
// r["d"]<<str;//double with string
// r<<"print(d)";

r["Function"]<<ROOT::R::TRFunctionExport(Function);
r<<"print(Function(-1))";

r<<"print(Function(1))";//division by zero producess Inf.

r<<"print('hello ')"<<std::string("print('world ')");
r["x"]=123;
r["y"]=321;

Int_t x;
x=r["x"];
std::cout<<x<<std::endl;

r["y"]>>x;
std::cout<<x<<std::endl;

r<<"mat<-matrix(c(1,2,3,4),nrow=2)";

TMatrixD mat(2,2);
r["mat"]>>mat;

r["m"]<<mat;

Double_t b;
Int_t aa;
TString str2;
r["a"]>>aa;
r["v"]>>v;
r["sv"]>>sv;
r["m"]>>m;
r["b"]>>b;
r["i"]>>i;
r["d"]>>d;
r["f"]>>f;
r["array"]>>a;
r["s"]>>str2;

mat.Print();
std::cout<<" array={"<<a[0]<<","<<a[1]<<","<<a[2]<<"}";
r["func"]<<Function;
r<<"print(func(2))";
std::cout<<"func="<<Function(2);

//passing overloaded functions
r["funi"]<<(Int_t (*)(Int_t))Fun;
r<<"print(funi(2))";
std::cout<<"funi="<<Fun(2)<<std::endl;

r["fund"]<<(Double_t (*)(Double_t))Fun;
r<<"print(fund(2.01))";
std::cout<<"fund="<<Fun(2.01)<<std::endl;

//if you uncomment the next line you get a big
//traceback because the template can not reslve the overloaded
//function.
//r["fun"]<<Fun;
}
