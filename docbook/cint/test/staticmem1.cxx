/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//*** Class Sample to perform statistics on various multi-dim. data samples
//*** NVE 30-mar-1996 CERN Geneva 
// A data sample can be filled using the "enter" and/or "remove" functions,
// whereas the "reset" function resets the complete sample to 'empty'.
// The info which can be extracted from a certain data sample are the
// sum, mean, variance, sigma, covariance and correlation.
// The "print" function provides all statistics data for a certain sample.
// The variables for which these stat. parameters have to be calculated
// are indicated by the index of the variable which is passed as an
// argument to the various member functions.
// The index convention for a data point (x,y) is : x=1  y=2
//
// Example :
// ---------
// For a Sample s a data point (x,y) can be entered as s.enter(x,y) and
// the mean_x can be obtained as s.mean(1) whereas the mean_y is obtained
// via s.mean(2). The correlation between x and y is available via s.cor(1,2).
// The x-statistics are obtained via s.print(1), y-statistics via s.print(2),
// and the covariance and correlation between x and y via s.print(1,2).
// All statistics of a sample are obtained via s.print().
//
#ifdef __hpux
#include <math.h>
#include <iostream.h>
#else
#include <cmath>
#include <iostream>
using namespace std;
#endif

class Sample
{
 public:
  Sample();
  void reset();
  void enter(float x);
  void remove(float x);
  void enter(float x, float y);
  void remove(float x, float y);
  void enter(float x, float y, float z);
  void remove(float x, float y, float z);
  int dim();
  int n();
  float sum(int i);
  float mean(int i);
  float var(int i);
  float sigma(int i);
  float cov(int i, int j);
  float cor(int i, int j);
  void print();
  void print(int i);
  void print(int i, int j);
 private:
  int the_dim; // Dimension of the sample
  int the_n;   // Number of entries of the sample
  enum {maxdim=3}; // Maximum supported dimension
  char the_names[maxdim]; // Variable names i.e. X,Y,Z
  float the_sum[maxdim];  // Total sum for each variable
  float the_sum2[maxdim][maxdim]; // Total sum**2 for each variable
  void calc();
  float the_mean[maxdim];  // Mean for each variable
  float the_var[maxdim];   // Variation for each variable
  float the_sigma[maxdim]; // Standard deviation for each variable
  float the_cov[maxdim][maxdim]; // Covariances of pairs of variables
  float the_cor[maxdim][maxdim]; // Correlations of pairs of variables
};

Sample::Sample()
{
//*** Creation of a Sample object and resetting the statistics values
//*** The dimension is initialised to maximum
 the_dim=maxdim;
 the_names[0]='X';
 the_names[1]='Y';
 the_names[2]='Z';
 the_n=0;
 for (int i=0; i<maxdim; i++)
 {
  the_sum[i]=0.;
  the_mean[i]=0.;
  the_var[i]=0.;
  the_sigma[i]=0.;
  for (int j=0; j<the_dim; j++)
  {
   the_sum2[i][j]=0.;
   the_cov[i][j]=0.;
   the_cor[i][j]=0.;
  }
 }
}

void Sample::reset()
{
//*** Resetting the statistics values for a certain Sample object
//*** Dimension is NOT changed
 the_n=0;
 for (int i=0; i<the_dim; i++)
 {
  the_sum[i]=0.;
  the_mean[i]=0.;
  the_var[i]=0.;
  the_sigma[i]=0.;
  for (int j=0; j<the_dim; j++)
  {
   the_sum2[i][j]=0.;
   the_cov[i][j]=0.;
   the_cor[i][j]=0.;
  }
 }
}

void Sample::enter(float x)
{
//*** Entering a value into a 1-dim. sample
//*** In case of first entry the dimension is set to 1
 if (the_n==0)
 {
  the_dim=1;
  the_names[0]='X';
  the_names[1]='-';
  the_names[2]='-';
 }
 if (the_dim != 1)
 {
  cout << " *Sample::enter* Error : Not a 1-dim sample." << endl;
 }
 else
 {
  the_n+=1;
  the_sum[0]+=x;
  the_sum2[0][0]+=x*x;
  calc();
 }
}

void Sample::remove(float x)
{
//*** Removing a value from a 1-dim. sample
 if (the_dim != 1)
 {
  cout << " *Sample::remove* Error : Not a 1-dim sample." << endl;
 }
 else
 {
  the_n-=1;
  the_sum[0]-=x;
  the_sum2[0][0]-=x*x;
  calc();
 }
}

void Sample::enter(float x, float y)
{
//*** Entering a pair (x,y) into a 2-dim. sample
//*** In case of first entry the dimension is set to 2
 if (the_n==0)
 {
  the_dim=2;
  the_names[0]='X';
  the_names[1]='Y';
  the_names[2]='-';
 }
 if (the_dim != 2)
 {
  cout << " *Sample::enter* Error : Not a 2-dim sample." << endl;
 }
 else
 {
  the_n+=1;
  the_sum[0]+=x;
  the_sum[1]+=y;
  the_sum2[0][0]+=x*x;
  the_sum2[0][1]+=x*y;
  the_sum2[1][0]+=y*x;
  the_sum2[1][1]+=y*y;
  calc();
 }
}

void Sample::remove(float x, float y)
{
//*** Removing a pair (x,y) from a 2-dim. sample
 if (the_dim != 2)
 {
  cout << " *Sample::remove* Error : Not a 2-dim sample." << endl;
 }
 else
 {
  the_n-=1;
  the_sum[0]-=x;
  the_sum[1]-=y;
  the_sum2[0][0]-=x*x;
  the_sum2[0][1]-=x*y;
  the_sum2[1][0]-=y*x;
  the_sum2[1][1]-=y*y;
  calc();
 }
}

void Sample::enter(float x, float y, float z)
{
//*** Entering a set (x,y,z) into a 3-dim. sample
//*** In case of first entry the dimension is set to 3
 if (the_n==0)
 {
  the_dim=3;
  the_names[0]='X';
  the_names[1]='Y';
  the_names[2]='Z';
 }
 if (the_dim != 3)
 {
  cout << " *Sample::enter* Error : Not a 3-dim sample." << endl;
 }
 else
 {
  the_n+=1;
  the_sum[0]+=x;
  the_sum[1]+=y;
  the_sum[2]+=z;
  the_sum2[0][0]+=x*x;
  the_sum2[0][1]+=x*y;
  the_sum2[0][2]+=x*z;
  the_sum2[1][0]+=y*x;
  the_sum2[1][1]+=y*y;
  the_sum2[1][2]+=y*z;
  the_sum2[2][0]+=z*x;
  the_sum2[2][1]+=z*y;
  the_sum2[2][2]+=z*z;
  calc();
 }
}

void Sample::remove(float x, float y, float z)
{
//*** Removing a set (x,y,z) from a 3-dim. sample
 if (the_dim != 3)
 {
  cout << " *Sample::remove* Error : Not a 3-dim sample." << endl;
 }
 else
 {
  the_n-=1;
  the_sum[0]-=x;
  the_sum[1]-=y;
  the_sum[2]-=z;
  the_sum2[0][0]-=x*x;
  the_sum2[0][1]-=x*y;
  the_sum2[0][2]-=x*z;
  the_sum2[1][0]-=y*x;
  the_sum2[1][1]-=y*y;
  the_sum2[1][2]-=y*z;
  the_sum2[2][0]-=z*x;
  the_sum2[2][1]-=z*y;
  the_sum2[2][2]-=z*z;
  calc();
 }
}

void Sample::calc()
{
//*** Calculation of the various statistical values
//*** after each entering or removing action on a certain sample
  int i;
  float rn=the_n;
  for (i=0; i<the_dim; i++) {
    the_mean[i]=the_sum[i]/rn;
    the_var[i]=(the_sum2[i][i]/rn)-(the_mean[i]*the_mean[i]);
    if (the_var[i] < 0.) the_var[i]=0.;
    the_sigma[i]=sqrt(the_var[i]);
  }
  float test;
  for (i=0; i<the_dim; i++) {
    for (int j=0; j<the_dim; j++) {
      the_cov[i][j]=(the_sum2[i][j]/rn)-(the_mean[i]*the_mean[j]);
      test=the_sigma[i]*the_sigma[j];
      if (test > 0.) the_cor[i][j]=the_cov[i][j]/test;
    }
  }
}

int Sample::dim()
{
//*** Provide the dimension of a certain sample
 return the_dim;
}

int Sample::n()
{
//*** Provide the number of entries of a certain sample
 return the_n;
}

float Sample::sum(int i)
{
//*** Provide the sum of a certain variable
  if (the_dim < i) {
    cout << " *Sample::sum* Error : Dimension less than " << i << endl;
    return the_sum[0];
  }
  else {
    return the_sum[i-1];
  }

}

float Sample::mean(int i)
{
//*** Provide the mean of a certain variable
 if (the_dim < i) {
   cout << " *Sample::mean* Error : Dimension less than " << i << endl;
   return the_mean[0];
 }
 else {
   return the_mean[i-1];
 }
}

float Sample::var(int i)
{
//*** Provide the variance of a certain variable
 if (the_dim < i) {
   cout << " *Sample::var* Error : Dimension less than " << i << endl;
   return the_var[0];
 }
 else {
   return the_var[i-1];
 }
}

float Sample::sigma(int i)
{
//*** Provide the standard deviation of a certain variable
  if (the_dim < i) {
    cout << " *Sample::sigma* Error : Dimension less than " << i << endl;
    return the_sigma[0];
  }
  else {
    return the_sigma[i-1];
  }
}

float Sample::cov(int i, int j)
{
//*** Provide the covariance between variables i and j
  if ((the_dim < i) || (the_dim < j)) {
    int k=i;
    if (j > i) k=j;
    cout << " *Sample::cov* Error : Dimension less than " << k << endl;
    return the_cov[0][0];
  }
  else {
    return the_cov[i-1][j-1];
  }
}

float Sample::cor(int i, int j)
{
//*** Provide the correlation between variables i and j
  if ((the_dim < i) || (the_dim < j)) {
    int k=i;
    if (j > i) k=j;
    cout << " *Sample::cor* Error : Dimension less than " << k << endl;
    return the_cor[0][0];
  }
  else {
    return the_cor[i-1][j-1];
  }
}


void Sample::print()
{
//*** Printing of statistics of all variables
  for (int i=0; i<the_dim; i++) {
    cout << " " << the_names[i] << " : N = " << the_n;
    cout << " Sum = " << the_sum[i] << " Mean = " << ( fabs(the_mean[i]) < 1e-7 ? 0.0 : the_mean[i] );
    cout << " Var = " << ( fabs(the_var[i]) < 1e-7 ? 0.0 : the_var[i] ) << " Sigma = " <<  ( fabs(the_sigma[i]) < 1e-7 ? 0.0 : the_sigma[i] ) << endl;
  }
}

void Sample::print(int i)
{
//*** Printing of statistics of ith variable
 if (the_dim < i) {
   cout << " *Sample::print(i)* Error : Dimension less than " << i << endl;
 }
 else {
   int oldprecision = cout.precision(3);
   cout << " " << the_names[i-1] << " : N = " << the_n;
   cout << " Sum = " << the_sum[i-1] << " Mean = " << ( fabs(the_mean[i-1]) < 1e-7 ? 0.0 : the_mean[i-1] );
   cout << " Var = " << ( fabs(the_var[i-1]) < 1e-7 ? 0.0 : the_var[i-1] ) << " Sigma = " << ( fabs(the_sigma[i-1])<1e-7 ? 0.0 : the_sigma[i-1] ) << endl;
   cout.precision(oldprecision);
 }
}

void Sample::print(int i, int j)
{
//*** Printing of covariance and correlation between variables i and j
 if ((the_dim < i) || (the_dim < j))
 {
  int k=i;
  if (j > i) k=j;
  cout << " *Sample::print(i,j)* Error : Dimension less than " << k << endl;
 }
 else
 {
  cout << " " << the_names[i-1] << "-" << the_names[j-1] << " correlation :";
  cout << " Cov. = " << the_cov[i-1][j-1] << " Cor. = " << the_cor[i-1][j-1] << endl;
 }
}


//*** The main program to test the various stat. functions

int main()
{
 Sample t;
 t.enter(9);
 cout << " Sample test :" << endl;
 t.print(1);
 Sample s1;
 s1.enter(19.);
 s1.enter(18.7);
 s1.enter(19.3);
 s1.enter(19.2);
 s1.enter(18.9);
 s1.enter(19.);
 s1.enter(20.2);
 s1.enter(19.9);
 s1.enter(18.6);
 s1.enter(19.4);
 s1.enter(19.3);
 s1.enter(18.8);
 s1.enter(19.3);
 s1.enter(19.2);
 s1.enter(18.7);
 s1.enter(18.5);
 s1.enter(18.6);
 s1.enter(19.7);
 s1.enter(19.9);
 s1.enter(20.);
 s1.enter(19.5);
 s1.enter(19.4);
 s1.enter(19.6);
 s1.enter(20.);
 s1.enter(18.9);
 cout << " Sample s1 students :" << endl;
 s1.print(1);
 s1.enter(37.);
 cout << " Sample s1 stud.+teacher :" << endl;
 s1.print(1);
 s1.remove(37.);
 cout << " Sample s1 teacher rem. :" << endl;
 s1.print(1);
 s1.reset();
 s1.enter(1);
 s1.enter(2.);
 int i=3;
 s1.enter(i);
 cout << " Sample s1 just some values :" << endl;
 s1.print(1);
 Sample s2;
 s2.enter(1,10);
 s2.enter(2,20);
 s2.enter(3,30);
 cout << " Sample s2 :" << endl;
 s2.print(1);
 s2.print(2);
 s2.reset();
 s2.enter(22,63);
 s2.enter(48,39);
 s2.enter(76,61);
 s2.enter(10,30);
 s2.enter(22,51);
 s2.enter(4,44);
 s2.enter(68,74);
 s2.enter(44,78);
 s2.enter(10,55);
 s2.enter(76,58);
 s2.enter(14,41);
 s2.enter(56,69);
 cout << " Sample s2 (CM,QM) :" << endl;
 s2.print(1);
 s2.print(2);
 s2.print(1,2);
 Sample s3;
 s3.enter(22,63,22);
 s3.enter(48,39,48);
 s3.enter(76,61,76);
 s3.enter(10,30,10);
 s3.enter(22,51,22);
 s3.enter(4,44,4);
 s3.enter(68,74,68);
 s3.enter(44,78,44);
 s3.enter(10,55,10);
 s3.enter(76,58,76);
 s3.enter(14,41,14);
 s3.enter(56,69,56);
 cout << " Sample s3 (CM,QM,CM) :" << endl;
 s3.print();
 s3.print(1,2);
 s3.print(1,3);

 return 0;
}
