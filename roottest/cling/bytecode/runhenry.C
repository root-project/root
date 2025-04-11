#include<vector>
#include<fstream>
#include<TGraph.h>
#include<TGraphErrors.h>
#include<TF1.h>
#include<TMath.h>
#include<TVirtualFitter.h>

using namespace std;
double CotTheta(const double x_i, const double alpha[2]){
  // returns the estimate y(x_i) based on the theoretical distribution
  return alpha[0]*x_i - alpha[1]*pow(x_i,-1);
}

double CalculateChi2(const vector<double> x, const vector<double> y, const vector<double> yError, const double alpha[2]){
  // returns the chi2 value for a given set of parameters
  double sumchi2=0;
  for (unsigned int i = 0; i<x.size(); i++){
    sumchi2=sumchi2+(pow(yError[i],-2)*pow(y[i]-CotTheta(x[i],alpha),2));
  }
  return sumchi2;
}

void runhenry(){

  TVirtualFitter::SetDefaultFitter("Minuit"); 

  // readout henry.dat
  vector<double> xVec;
  vector<double> xEVec;
  vector<double> yVec;
  vector<double> yEVec;
  double x(0),y(0),xE(0), yE(0);
  double output[3];
  FILE *data;
  char datafname[]="henry.dat";
  
  data = fopen(datafname,"r");
  while (fscanf(data, "%lf %lf %lf %lf",&x,&xE,&y,&yE) != EOF) {
    xVec.push_back(x);
    yVec.push_back(y);
    xEVec.push_back(xE);
    yEVec.push_back(yE);
  }
  


  // my fit routine...
   double alpha[2];
   double chi2;
   alpha[0]=1e-4;
   alpha[1]=1e2;
   // single call works just fine...
   chi2 = CalculateChi2(xVec,yVec,yEVec,alpha);
   cout << "initial chi2/Ndf: " << chi2/xVec.size() << endl;
   // same call but inside a loop doesn't work...
   for (int i = 0; i<10; i++){
     chi2=CalculateChi2(xVec,yVec,yEVec,alpha);
     cout << "Chi2/Ndf " << chi2/xVec.size() << endl;
   }
}
