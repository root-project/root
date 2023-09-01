// stress inversion of matrices with varios methods
#include "Math/SMatrix.h"
#include "TMatrixTSym.h"
#include "TDecompChol.h"
#include "TDecompBK.h"

#include "TRandom.h"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <limits>

#include "TStopwatch.h"

// matrix size
constexpr unsigned int N = 5;

bool doSelfTest = true;

//timer
namespace test {


#ifdef REPORT_TIME
   void reportTime( std::string s, double time);
#endif

   void printTime(TStopwatch & time, std::string s) {
      int pr = std::cout.precision(8);
      std::cout << s << "\t" << " time = " << time.RealTime() << "\t(sec)\t"
         //    << time.CpuTime()
                << std::endl;
      std::cout.precision(pr);
   }



   class Timer {

   public:

      Timer(const std::string & s = "") : fName(s), fTime(0)
      {
         fWatch.Start();
      }
      Timer(double & t, const std::string & s = "") : fName(s), fTime(&t)
      {
         fWatch.Start();
      }

      ~Timer() {
         fWatch.Stop();
         printTime(fWatch,fName);
#ifdef REPORT_TIME
         // report time
         reportTime(fName, fWatch.RealTime() );
#endif
         if (fTime) *fTime += fWatch.RealTime();
      }


   private:

      std::string fName;
      double * fTime;
      TStopwatch fWatch;

   };
}

using namespace ROOT::Math;



typedef  SMatrix<double,N,N, MatRepSym<double,N> >  SymMatrix;

// create matrix
template<class M>
M * createMatrix() {
   return new M();
}
// specialized for TMatrix
template<>
TMatrixTSym<double> * createMatrix<TMatrixTSym<double> >() {
   return new TMatrixTSym<double>(N);
}

//print matrix
template<class M>
void printMatrix(const M & m) {
   std::cout << m << std::endl;
}
template<>
void printMatrix<TMatrixTSym<double> >(const TMatrixTSym<double> & m ) {
   m.Print();
}

// generate matrices
template<class M>
void genMatrix(M  & m ) {
   TRandom & r = *gRandom;
   // generate first diagonal elemets
   for (int i = 0; i < N; ++i) {
      double maxVal = i*10000/(N-1) + 1;  // max condition is 10^4
      m(i,i) = r.Uniform(0, maxVal);
   }
   for (int i = 0; i < N; ++i) {
      for (int j = 0; j < i; ++j) {
         double v = 0.3*std::sqrt( m(i,i) * m(j,j) ); // this makes the matrix pos defined
         m(i,j) = r.Uniform(0, v);
         m(j,i) = m(i,j); // needed for TMatrix
      }
   }
}

// generate all matrices
template<class M>
void generate(std::vector<M*> & v) {
   int n = v.size();
   gRandom->SetSeed(111);
   for (int i = 0; i < n; ++i) {
      v[i] = createMatrix<M>();
      genMatrix(*v[i] );
   }
}


struct Choleski {};
struct BK {};
struct QR {};
struct Cramer {};
struct Default {};

template <class M, class Type>
struct TestInverter {
   static bool Inv ( const M & , M & ) { return false;}
   static bool Inv2 ( M & ) { return false;}
};

template <>
struct TestInverter<SymMatrix, Choleski> {
   static bool Inv ( const SymMatrix & m, SymMatrix & result ) {
      int ifail = 0;
      result = m.InverseChol(ifail);
      return  ifail == 0;
   }
   static bool Inv2 ( SymMatrix & m ) {
      return m.InvertChol();
   }
};

template <>
struct TestInverter<SymMatrix, BK> {
   static bool Inv ( const SymMatrix & m, SymMatrix & result ) {
      int ifail = 0;
      result = m.Inverse(ifail);
      return ifail==0;
   }
   static bool Inv2 ( SymMatrix & m ) {
      return m.Invert();
   }
};

template <>
struct TestInverter<SymMatrix, Cramer> {
   static bool Inv ( const SymMatrix & m, SymMatrix & result ) {
      int ifail = 0;
      result = m.InverseFast(ifail);
      return ifail==0;
   }
   static bool Inv2 ( SymMatrix & m ) {
      return m.InvertFast();
   }
};

#ifdef LATER
template <>
struct TestInverter<SymMatrix, QR> {
   static bool Inv ( const SymMatrix & m, SymMatrix & result ) {
      ROOT::Math::QRDecomposition<double> d;
      int ifail = 0;
      result = m.InverseFast(ifail);
      return ifail==0;
   }
};
#endif

//TMatrix functions

template <>
struct TestInverter<TMatrixDSym, Default> {
   static bool Inv ( const TMatrixDSym & m, TMatrixDSym & result ) {
      result = m;
      result.Invert();
      return true;
   }
   static bool Inv2 ( TMatrixDSym & m ) {
      m.Invert();
      return true;
   }
};

template <>
struct TestInverter<TMatrixDSym, Cramer> {
   static bool Inv ( const TMatrixDSym & m, TMatrixDSym & result ) {
      result = m;
      result.InvertFast();
      return true;
   }
   static bool Inv2 ( TMatrixDSym & m ) {
      m.InvertFast();
      return true;
   }
};

template <>
struct TestInverter<TMatrixDSym, Choleski> {
   static bool Inv ( const TMatrixDSym & m, TMatrixDSym & result ) {
      TDecompChol chol(m);
      if (!chol.Decompose() ) return false;
      chol.Invert(result);
      return true;
   }
};

template <>
struct TestInverter<TMatrixDSym, BK> {
   static bool Inv ( const TMatrixDSym & m, TMatrixDSym & result ) {
      TDecompBK d(m);
      if (!d.Decompose() ) return false;
      d.Invert(result);
      return true;
   }
};


template<class M, class T>
double invert( const std::vector<M* >  & matlist, double & time,std::string s) {
   M result = *(matlist.front());
   test::Timer t(time,s);
   int nloop = matlist.size();
   double sum = 0;
   for (int l = 0; l < nloop; l++)
   {
      const M & m = *(matlist[l]);
      bool ok = TestInverter<M,T>::Inv(m,result);
      if (!ok) {
         std::cout << "inv failed for matrix " << l << std::endl;
         printMatrix<M>( m);
         return -1;
      }
      sum += result(0,1);
   }
   return sum;
}

// invert without copying the matrices (une INv2)

template<class M, class T>
double invert2( const std::vector<M* >  & matlist, double & time,std::string s) {

   // copy vector of matrices
   int nloop = matlist.size();
   std::vector<M *> vmat(nloop);
   for (int l = 0; l < nloop; l++)
   {
      vmat[l] = new M( *matlist[l] );
   }

   test::Timer t(time,s);
   double sum = 0;
   for (int l = 0; l < nloop; l++)
   {
      M & m = *(vmat[l]);
      bool ok = TestInverter<M,T>::Inv2(m);
      if (!ok) {
         std::cout << "inv failed for matrix " << l << std::endl;
         printMatrix<M>( m);
         return -1;
      }
      sum += m(0,1);
   }
   return sum;
}

bool equal(double d1, double d2, double stol = 10000) {
   std::cout.precision(12);  // tolerance is 1E-12
   double eps = stol * std::numeric_limits<double>::epsilon();
   if ( std::abs(d1) > 0 && std::abs(d2) > 0 )
      return  ( std::abs( d1-d2) < eps * std::max(std::abs(d1), std::abs(d2) ) );
   else if ( d1 == 0 )
      return std::abs(d2) < eps;
   else // d2 = 0
      return std::abs(d1) < eps;
}

// test matrices symmetric and positive defines
bool stressSymPosInversion(int n, bool selftest ) {

   // test smatrix

   std::vector<SymMatrix *> v1(n);
   generate(v1);
   std::vector<TMatrixDSym *> v2(n);
   generate(v2);


   bool iret = true;
   double time = 0;
   double s1 = invert<SymMatrix, Choleski> (v1, time,"SMatrix Chol");
   double s2 = invert<SymMatrix, BK> (v1, time,"SMatrix   BK");
   double s3 = invert<SymMatrix, Cramer> (v1, time,"SMatrix Cram");
   bool ok = ( equal(s1,s2) && equal(s1,s3) );
   if (!ok) {
      std::cout << "result SMatrix choleski  " << s1 << " BK   " << s2 << " cramer " << s3 << std::endl;
      std::cerr <<"Error:  inversion test for SMatrix FAILED ! " << std::endl;
   }
   iret  &= ok;
   std::cout << std::endl;

   double m1 = invert<TMatrixDSym, Choleski> (v2, time,"TMatrix Chol");
   double m2 = invert<TMatrixDSym, BK> (v2, time,"TMatrix   BK");
   double m3 = invert<TMatrixDSym, Cramer> (v2, time,"TMatrix Cram");
   double m4 = invert<TMatrixDSym, Default> (v2, time,"TMatrix  Def");

   ok =  ( equal(m1,m2) && equal(m1,m3) && equal(m1,m4) );
   if (!ok) {
      std::cout << "result TMatrix choleski  " << m1 << " BK   " << m2
                << " cramer " << m3 << " default " << m4 << std::endl;
      std::cerr <<"Error:  inversion test for TMatrix FAILED ! " << std::endl;
   }
   iret  &= ok;
   std::cout << std::endl;


      // test using self inversion
   if (selftest) {
      std::cout << "\n - self inversion test \n";
      double s11 = invert2<SymMatrix, Choleski> (v1, time,"SMatrix Chol");
      double s12 = invert2<SymMatrix, BK> (v1, time,"SMatrix   BK");
      double s13 = invert2<SymMatrix, Cramer> (v1, time,"SMatrix Cram");
      ok =  ( equal(s11,s12) && equal(s11,s13) );
      if (!ok) {
         std::cout << "result SMatrix choleski  " << s11 << " BK   " << s12 << " cramer " << s13 << std::endl;
         std::cerr <<"Error:  self inversion test for SMatrix FAILED ! " << std::endl;
      }
      iret  &= ok;
      std::cout << std::endl;

      double m13 = invert2<TMatrixDSym, Cramer> (v2, time,"TMatrix Cram");
      double m14 = invert2<TMatrixDSym, Default> (v2, time,"TMatrix  Def");
      ok =  ( equal(m13,m14)  );
      if (!ok) {
         std::cout << "result TMatrix  cramer " << m13 << " default " << m14 << std::endl;
         std::cerr <<"Error:  self inversion test for TMatrix FAILED ! " << std::endl;
      }
      iret  &= ok;
      std::cout << std::endl;
   }

   return iret;
}

int testInversion(int n = 100000) {
   std::cout << "Test Inversion for matrix with N = " << N << std::endl;
   bool ok = stressSymPosInversion(n, doSelfTest);
   std::cerr << "Test inversion of positive defined matrix ....... ";
   if (ok) std::cerr << "OK \n";
   else std::cerr << "FAILED \n";
   return (ok) ? 0 : -1;
}

int main() {
   return testInversion();
}
