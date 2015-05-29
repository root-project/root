//Example based  in
//http://cran.r-project.org/web/packages/DEoptim/DEoptim.pdf
//Please install the R package DEoptim before run this example.
//Author: Omar Zapata

#include<TRInterface.h>
#include<TBenchmark.h>
#include<math.h>
#include<stdlib.h>
//In the next function the *double pointer should be changed by a TVectorD datatype,
//because the pointer has no meaning in R's enviroment.
//This is a generalization of the RosenBrock function, with the min xi=1 and i>0.
Double_t GenRosenBrock(const TVectorD xx )
{
  int length=xx.GetNoElements();
  
  Double_t result=0;
  for(int i=0;i<(length-1);i++)
  {
    result+=pow(1-xx[i],2)+100*pow(xx[i+1]-pow(xx[i],2),2);
  }
  return result;
}

//the min xi=0 i>0
Double_t Rastrigin(const TVectorD xx)
{
  int length=xx.GetNoElements();
  Double_t result=10*length;
  for(int i=0;i<length;i++)
  {
    result+=xx[i]*xx[i]-10*cos(6.2831853*xx[i]);
  }
  return result;
}

void GlobalMinimization()
{
 TBenchmark bench;
 ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
 
 Bool_t installed=r.Eval("is.element('DEoptim', installed.packages()[,1])");
 if(!installed)
 {
    std::cout<<"Package DEoptim no installed in R"<<std::endl;
    std::cout<<"Run install.packages('DEoptim') in R's environment"<<std::endl;
    return;
 }
 
 //loading DEoptim
 r<<"suppressMessages(library(DEoptim, quietly = TRUE))";
 
//  passing RosenBrock function to R
 r["GenRosenBrock"]<<ROOT::R::TRFunction(GenRosenBrock);

 //maximun number of iterations 
 r["MaxIter"]<<5000;
 //n = size of vector that is an argument for GenRosenBrock
 r["n"]<<3;
 //lower limits
 r<<"ll<-rep(-25, n)";
 //upper limits
 r<<"ul<-rep(25, n)";
 
 bench.Start("GlobalMinimizationRosenBrock");
 //calling minimization and timing it.
 r<<"result1<-DEoptim(fn=GenRosenBrock,lower=ll,upper=ul,control=list(NP=10*n,itermax=MaxIter,trace=FALSE))";
 std::cout<<"-----------------------------------------"<<std::endl;
 std::cout<<"RosenBrock's minimum in: "<<std::endl;
 r<<"print(result1$optim$bestmem)";
 std::cout<<"Bechmark Times"<<std::endl;
//  printing times
 bench.Show("GlobalMinimizationRosenBrock");

 
 //passing RosenBrock function to R
 r["Rastrigin"]<<ROOT::R::TRFunction(Rastrigin);
 //maximun number of iterations 
 r["MaxIter"]<<2000;
 //n = size of a vector which is an argument for Rastrigin
 r["n"]<<3;
 //lower limits
 r<<"ll<-rep(-5, n)";
 //upper limits
 r<<"ul<-rep(5, n)";
 
 bench.Start("GlobalMinimizationRastrigin");
 //calling minimization and timing it.
 r<<"result2<-DEoptim(fn=Rastrigin,lower=ll,upper=ul,control=list(NP=10*n,itermax=MaxIter,trace=FALSE))";
 std::cout<<"-----------------------------------------"<<std::endl;
 std::cout<<"Rastrigin's minimum in: "<<std::endl;
 r<<"print(result2$optim$bestmem)";
 std::cout<<"Bechmark Times"<<std::endl;
 //printing times
 bench.Show("GlobalMinimizationRastrigin");
 // skip R plotting in batch mode
 if (!gROOT->IsBatch()) {
    r<<"dev.new(title='RosenBrock Convergence')";
    r<<"plot(result1,type='o',pch='.')";
    r<<"dev.new(title='Rastrigin Convergence')";
    r<<"plot(result2,type='o',pch='.')";
 }
}
