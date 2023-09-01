
#define RUN_ALL_POINTS
//#define HAVE_CLHEP
//#define DEBUG

#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include "TMatrixD.h"
#include "TVectorD.h"

#include "TFile.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TRandom3.h"
#include "TStopwatch.h"

#include <iostream>

#include <map>

//#undef HAVE_CLHEP

#ifdef HAVE_CLHEP
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#endif


#define NITER 1  // number of iterations

#define NLOOP 1000000 // number of time the test is repeted



template <unsigned int NDIM1, unsigned int NDIM2>
class TestRunner {

public:

   // kalman test with normal  matrices from SMatrix
   int test_smatrix_kalman() ;
   // kalman test with sym   matrices from SMatrix
   int test_smatrix_sym_kalman();

   // kalman test with matrices (normal and sym) from TMatrix
   int test_tmatrix_kalman();

#ifdef HAVE_CLHEP
   // test with CLHEP matrix
   int test_clhep_kalman();
#endif

   // run the all tests
   int  run() {
      int iret = 0;
      iret |= test_smatrix_sym_kalman();
      iret |= test_smatrix_kalman();
      iret |= test_tmatrix_kalman();
#ifdef HAVE_CLHEP
      iret |= test_clhep_kalman();
#endif
      std::cout << "\n\n";

      return iret;
   }

private:
   double fX2sum;
   double fC2sum;

};


#define TR(N1,N2) \
{ TestRunner<N1,N2> tr; if (tr.run()) return -1; }


int runTest() {

#ifndef RUN_ALL_POINTS
   TR(2,5)
   //TR(2,10)
#else
   TR(2,2)
   TR(2,3)
   TR(2,4)
   TR(2,5)
   TR(2,6)
   TR(2,7)
   TR(2,8)
   TR(3,2)
   TR(3,3)
   TR(3,4)
   TR(3,5)
   TR(3,6)
   TR(3,7)
   TR(3,8)
   TR(4,2)
   TR(4,3)
   TR(4,4)
   TR(4,5)
   TR(4,6)
   TR(4,7)
   TR(4,8)
   TR(5,2)
   TR(5,3)
   TR(5,4)
   TR(5,5)
   TR(5,6)
   TR(5,7)
   TR(5,8)
   TR(6,2)
   TR(6,3)
   TR(6,4)
   TR(6,5)
   TR(6,6)
   TR(6,7)
   TR(6,8)
   TR(7,2)
   TR(7,3)
   TR(7,4)
   TR(7,5)
   TR(7,6)
   TR(7,7)
   TR(7,8)
   TR(8,2)
   TR(8,3)
   TR(8,4)
   TR(8,5)
   TR(8,6)
   TR(8,7)
   TR(8,8)
   TR(9,2)
   TR(9,3)
   TR(9,4)
   TR(9,5)
   TR(9,6)
   TR(9,7)
   TR(9,8)
   TR(10,2)
   TR(10,3)
   TR(10,4)
   TR(10,5)
   TR(10,6)
   TR(10,7)
   TR(10,8)
#endif
   return 0;
}



// utility functions to fill matrices and vectors with random data

template<class V>
void fillRandomVec(TRandom & r, V  & v, unsigned int len, unsigned int start = 0, double offset = 1) {
   for(unsigned int i = start; i < len+start; ++i)
      v[i] = r.Rndm() + offset;
}

template<class M>
void fillRandomMat(TRandom & r, M  & m, unsigned int first, unsigned int second, unsigned int start = 0, double offset = 1) {
   for(unsigned int i = start; i < first+start; ++i)
      for(unsigned int j = start; j < second+start; ++j)
         m(i,j) = r.Rndm() + offset;
}

template<class M>
void fillRandomSym(TRandom & r, M  & m, unsigned int first, unsigned int start = 0, double offset = 1) {
   for(unsigned int i = start; i < first+start; ++i) {
      for(unsigned int j = i; j < first+start; ++j) {
         if ( i != j ) {
            m(i,j) = r.Rndm() + offset;
            m(j,i) = m(i,j);
         }
         else // add extra offset to make no singular when inverting
            m(i,i) = r.Rndm() + 3*offset;
      }
   }
}



// simple class to measure time


void printTime(TStopwatch & time, std::string s) {
   int pr = std::cout.precision(4);
   std::cout << std::setw(12) << s << "\t" << " Real time = " << time.RealTime() << "\t(sec)\tCPU time = "
   << time.CpuTime() << "\t(sec)"
   << std::endl;
   std::cout.precision(pr);
}

// reference times for sizes <=6 and > 6 on Linux slc3 P4 3Ghz ("SMatrix","SMatrix_sym","TMatrix")
double refTime1[4] = { 40.49, 53.75, 83.21,1000 };
double refTime2[4] = { 393.81, 462.16, 785.50,10000 };

#define NMAX1  9   // matrix storese results from 2 to 10
#define NMAX2  7   //  results from 2 to 8
                   // class to hold time results
class TimeReport {

   //  typedef  std::map<std::string, ROOT::Math::SMatrix<double,NMAX1,NMAX2,ROOT::Math::MatRepSym<double,NMAX1,NMAX2> > > ResultTable;
   typedef std::map<std::string, double> a;
   typedef   ROOT::Math::SMatrix<double,NMAX1,NMAX2> M;
   typedef  std::map< std::string, M > ResultTable;

public:

   TimeReport() {}

   ~TimeReport() { /*print(); */   }

   // set timing point
   void Set(const std::string & name, int dim1, int dim2 ) {
      fDim1 = dim1;
      fDim2 = dim2;
      fName = name;
      // insert in map if not existing
      if (fResult1.find(name) == fResult1.end() )
         fResult1.insert(ResultTable::value_type(name, M() ) );
      if (fResult2.find(name) == fResult2.end() )
         fResult2.insert(ResultTable::value_type(name, M() )  );

   }

   std::string name() const { return fName; }

   void report(double rt, double ct) {
      fResult1[fName](fDim1-2,fDim2-2) = rt;
      fResult2[fName](fDim1-2,fDim2-2) = ct;
   }

   double smallSum(const M & m, int cut = 6) {
      // sum for sizes <= cut
      double sum = 0;
      for (int i = 0; i<cut-1; ++i)
         for (int j = 0; j<cut-1; ++j)
            sum+= m(i,j);

      return sum;
   }

   double largeSum(const M & m, int cut = 6) {
      // sum for sizes > cut
      double sum = 0;
      for (int i = 0; i<M::kRows; ++i)
         for (int j = 0; j<M::kCols; ++j)
            if ( i > cut-2 || j > cut-2)
               sum+= m(i,j);

      return sum;
   }

   void print(std::ostream & os) {
      std::map<std::string,double> r1;
      std::map<std::string,double> r2;
      os << "Real time results " << std::endl;
      for (ResultTable::iterator itr = fResult1.begin(); itr != fResult1.end(); ++itr) {
         std::string type = itr->first;
         os << " Results for " << type << std::endl;
         os << "\n" << itr->second << std::endl << std::endl;
         r1[type] = smallSum(itr->second);
         r2[type] = largeSum(itr->second);
         os << "\nTotal for N1,N2 <= 6    :  " << r1[type] << std::endl;
         os << "\nTotal for N1,N2 >  6    :  " << r2[type] << std::endl;
      }
      os << "\n\nCPU time results " << std::endl;
      for (ResultTable::iterator itr = fResult2.begin(); itr != fResult2.end(); ++itr) {
         os << " Results for " << itr->first << std::endl;
         os << "\n" << itr->second << std::endl << std::endl;
         os << "\nTotal for N1,N2 <= 6    :  " << smallSum(itr->second) << std::endl;
         os << "\nTotal for N1,N2 >  6    :  " << largeSum(itr->second) << std::endl;
      }

      // print summary
      os << "\n\n****************************************************************************\n";
      os << "Root version: " << gROOT->GetVersion() <<  "   "
      << gROOT->GetVersionDate() << "/" << gROOT->GetVersionTime() << std::endl;
      os <<     "****************************************************************************\n";
      os << "\n\t ROOTMARKS for N1,N2 <= 6 \n\n";
      int j = 0;
      os.setf(std::ios::right,std::ios::adjustfield);
      for (std::map<std::string,double>::iterator i = r1.begin(); i != r1.end(); ++i) {
         std::string type = i->first;
         os << std::setw(12) << type << "\t=\t" << refTime1[j]*800/r1[type] << std::endl;
         j++;
      }
      os << "\n\t ROOTMARKS for N1,N2 >  6 \n\n";
      j = 0;
      for (std::map<std::string,double>::iterator i = r1.begin(); i != r1.end(); ++i) {
         std::string type = i->first;
         os << std::setw(12) << type << "\t=\t" << refTime2[j]*800/r2[type] << std::endl;
         j++;
      }

   }

   void save(const std::string & fileName) {
      TFile file(fileName.c_str(),"RECREATE");
      gSystem->Load("libSmatrix");

      // save RealTime results
      for (ResultTable::iterator itr = fResult1.begin(); itr != fResult1.end(); ++itr) {
         int ret = file.WriteObject(&(itr->second),(itr->first).c_str() );
         if (ret ==0) std::cerr << "==> Error saving results in ROOT file " << fileName << std::endl;
      }

      // save CPU time results
      for (ResultTable::iterator itr = fResult2.begin(); itr != fResult2.end(); ++itr) {
         std::string typeName = itr->first + "_2";
         int ret = file.WriteObject(&(itr->second), typeName.c_str() );
         if (ret ==0) std::cerr << "==> Error saving results in ROOT file " << fileName << std::endl;
      }
      file.Close();
   }

private:

   int fDim1;
   int fDim2;
   std::string fName;

   ResultTable fResult1;
   ResultTable fResult2;

};

//global instance of time report
TimeReport gReporter;

class TestTimer {

public:

   // TestTimer(const std::string & s = "") :
   //     fName(s), fTime(0), fRep(0)
   //   {
   //     fWatch.Start();
   //   }
   TestTimer(TimeReport & r ) :
   fTime(0), fRep(&r)
   {
      fName = fRep->name();
      fWatch.Start();
   }
   TestTimer(double & t, const std::string & s = "") : fName(s), fTime(&t), fRep(0)
   {
      fWatch.Start();
   }

   ~TestTimer() {
      fWatch.Stop();
      printTime(fWatch,fName);
      if (fRep) fRep->report( fWatch.RealTime(), fWatch.CpuTime() );
      if (fTime) *fTime += fWatch.RealTime();
   }


private:

   std::string fName;
   double * fTime;
   TStopwatch fWatch;
   TimeReport * fRep;

};



template <unsigned int NDIM1, unsigned int NDIM2>
int TestRunner<NDIM1,NDIM2>::test_smatrix_kalman() {

   gReporter.Set("SMatrix",NDIM1,NDIM2);

   // need to write explicitly the dimensions


   typedef ROOT::Math::SMatrix<double, NDIM1, NDIM1>  MnMatrixNN;
   typedef ROOT::Math::SMatrix<double, NDIM2, NDIM2>  MnMatrixMM;
   typedef ROOT::Math::SMatrix<double, NDIM1, NDIM2>  MnMatrixNM;
   typedef ROOT::Math::SMatrix<double, NDIM2 , NDIM1> MnMatrixMN;
   typedef ROOT::Math::SMatrix<double, NDIM1 >        MnSymMatrixNN;
   typedef ROOT::Math::SMatrix<double, NDIM2 >        MnSymMatrixMM;
   typedef ROOT::Math::SVector<double, NDIM1>         MnVectorN;
   typedef ROOT::Math::SVector<double, NDIM2>         MnVectorM;



   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "****************************************************************************\n";
   std::cout << "\t\tSMatrix kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "****************************************************************************\n";




   int npass = NITER;
   TRandom3 r(111);
   double x2sum = 0, c2sum = 0;
   for (int k = 0; k < npass; k++) {



      MnMatrixNM H;
      MnMatrixMN K0;
      MnSymMatrixMM Cp;
      MnSymMatrixNN V;
      MnVectorN m;
      MnVectorM xp;


      {
         // fill matrices with random data
         fillRandomMat(r,H,first,second);
         fillRandomMat(r,K0,second,first);
         fillRandomSym(r,Cp,second);
         fillRandomSym(r,V,first);
         fillRandomVec(r,m,first);
         fillRandomVec(r,xp,second);
      }


      MnSymMatrixMM I;
      for(int i = 0; i < second; i++)
         I(i,i) = 1;

#ifdef DEBUG
      std::cout << "pass " << k << std::endl;
      if (k == 0) {
         std::cout << " K0 = " << K0 << std::endl;
         std::cout << " H = " << K0 << std::endl;
         std::cout << " Cp = " << Cp << std::endl;
         std::cout << " V = " << V << std::endl;
         std::cout << " m = " << m << std::endl;
         std::cout << " xp = " << xp << std::endl;
      }
#endif


      {
         double x2 = 0,c2 = 0;
         TestTimer t(gReporter);

         MnVectorM x;
         MnMatrixMN tmp;
         MnSymMatrixNN Rinv;
         MnMatrixMN K;
         MnSymMatrixMM C;
         MnVectorN vtmp1;
         MnVectorN vtmp;

         for (int l = 0; l < NLOOP; l++)
         {



            vtmp1 = H*xp -m;
            //x = xp + K0 * (m- H * xp);
            x = xp - K0 * vtmp1;
            tmp = Cp * Transpose(H);
            Rinv = V;  Rinv +=  H * tmp;

            bool test = Rinv.Invert();
            if(!test) {
               std::cout<<"inversion failed" <<std::endl;
               std::cout << Rinv << std::endl;
            }

            K =  tmp * Rinv ;
            C = Cp; C -= K * Transpose(tmp);
            //C = ( I - K * H ) * Cp;
            //x2 = Product(Rinv,m-H*xp);  // this does not compile on WIN32
            vtmp = m-H*xp;
            x2 = Dot(vtmp, Rinv*vtmp);

#ifdef DEBUG
            if (l == 0) {
               std::cout << " Rinv =\n " << Rinv << std::endl;
               std::cout << " C =\n " << C << std::endl;
            }
#endif

         }
         //std::cout << k << " chi2 = " << x2 << std::endl;
         x2sum += x2;
         c2 = 0;
         for (unsigned int i=0; i<NDIM2; ++i)
            for (unsigned int j=0; j<NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }
   }
   //tr.dump();

   int iret = 0;
   double d = std::abs(x2sum-fX2sum);
   if ( d > 1.E-6 * fX2sum  ) {
      std::cout << "ERROR: difference found in x2sum = " << x2sum << "\tref = " << fX2sum <<
      "\tdiff = " << d <<  std::endl;
      iret = 1;
   }
   d = std::abs(c2sum-fC2sum);
   if ( d > 1.E-6 * fC2sum  ) {
      std::cout << "ERROR: difference found in c2sum = " << c2sum << "\tref = " << fC2sum <<
      "\tdiff = " << d <<  std::endl;
      iret = 1;
   }

   return iret;
}

template <unsigned int NDIM1, unsigned int NDIM2>
int TestRunner<NDIM1,NDIM2>::test_smatrix_sym_kalman() {

   gReporter.Set("SMatrix_sym",NDIM1,NDIM2);

   // need to write explicitly the dimensions


   typedef ROOT::Math::SMatrix<double, NDIM1, NDIM1>  MnMatrixNN;
   typedef ROOT::Math::SMatrix<double, NDIM2, NDIM2>  MnMatrixMM;
   typedef ROOT::Math::SMatrix<double, NDIM1, NDIM2>  MnMatrixNM;
   typedef ROOT::Math::SMatrix<double, NDIM2 , NDIM1> MnMatrixMN;
   typedef ROOT::Math::SMatrix<double, NDIM1, NDIM1, ROOT::Math::MatRepSym<double, NDIM1> >   MnSymMatrixNN;
   typedef ROOT::Math::SMatrix<double, NDIM2, NDIM2, ROOT::Math::MatRepSym<double, NDIM2> >   MnSymMatrixMM;
   typedef ROOT::Math::SVector<double, NDIM1>         MnVectorN;
   typedef ROOT::Math::SVector<double, NDIM2>         MnVectorM;
   typedef ROOT::Math::SVector<double, NDIM1*(NDIM1+1)/2>   MnVectorN2;
   typedef ROOT::Math::SVector<double, NDIM2*(NDIM2+1)/2>   MnVectorM2;



   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "****************************************************************************\n";
   std::cout << "\t\tSMatrix_Sym kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "****************************************************************************\n";




   int npass = NITER;
   TRandom3 r(111);
   double x2sum = 0, c2sum = 0;
   for (int k = 0; k < npass; k++) {



      MnMatrixNM H;
      MnMatrixMN K0;
      MnSymMatrixMM Cp;
      MnSymMatrixNN V;
      MnVectorN m;
      MnVectorM xp;


      {
         // fill matrices with random data
         fillRandomMat(r,H,first,second);
         fillRandomMat(r,K0,second,first);
         fillRandomSym(r,Cp,second);
         fillRandomSym(r,V,first);
         fillRandomVec(r,m,first);
         fillRandomVec(r,xp,second);
      }


      MnSymMatrixMM I;
      for(int i = 0; i < second; i++)
         I(i,i) = 1;

#ifdef DEBUG
      std::cout << "pass " << k << std::endl;
      if (k == 0) {
         std::cout << " K0 = " << K0 << std::endl;
         std::cout << " H = " << K0 << std::endl;
         std::cout << " Cp = " << Cp << std::endl;
         std::cout << " V = " << V << std::endl;
         std::cout << " m = " << m << std::endl;
         std::cout << " xp = " << xp << std::endl;
      }

#endif

      {
         double x2 = 0,c2 = 0;
         TestTimer t(gReporter);

         MnVectorM x;
         MnSymMatrixNN RinvSym;
         MnMatrixMN K;
         MnSymMatrixMM C;
         MnSymMatrixMM Ctmp;
         MnVectorN vtmp1;
         MnVectorN vtmp;
#define OPTIMIZED_SMATRIX_SYM
#ifdef OPTIMIZED_SMATRIX_SYM
         MnMatrixMN tmp;
#endif

         for (int l = 0; l < NLOOP; l++)
         {


#ifdef OPTIMIZED_SMATRIX_SYM
            vtmp1 = H*xp -m;
            //x = xp + K0 * (m- H * xp);
            x = xp - K0 * vtmp1;
            tmp = Cp * Transpose(H);
            // we are sure that H*tmp result is symmetric
            ROOT::Math::AssignSym::Evaluate(RinvSym,H*tmp);
            RinvSym += V;

            bool test = RinvSym.Invert();
            if(!test) {
               std::cout<<"inversion failed" <<std::endl;
               std::cout << RinvSym << std::endl;
            }

            K =  tmp * RinvSym ;
            // we profit from the fact that result of K*tmpT is symmetric
            ROOT::Math::AssignSym::Evaluate(Ctmp, K*Transpose(tmp) );
            C = Cp; C -= Ctmp;
            //C = ( I - K * H ) * Cp;
            //x2 = Product(Rinv,m-H*xp);  // this does not compile on WIN32
            vtmp = m-H*xp;
            x2 = ROOT::Math::Dot(vtmp, RinvSym*vtmp);
#else
            // use similarity function
            vtmp1 = H*xp -m;
            x = xp - K0 * vtmp1;
            RinvSym = V;  RinvSym +=  Similarity(H,Cp);

            bool test = RinvSym.Invert();
            if(!test) {
               std::cout<<"inversion failed" <<std::endl;
               std::cout << RinvSym << std::endl;
            }

            Ctmp = ROOT::Math::SimilarityT(H, RinvSym);
            C = Cp; C -= ROOT::Math::Similarity(Cp, Ctmp);
            vtmp = m-H*xp;
            x2 = ROOT::Math::Similarity(vtmp, RinvSym);
#endif

         }
         //std::cout << k << " chi2 = " << x2 << std::endl;
         x2sum += x2;
         c2 = 0;
         for (unsigned int i=0; i<NDIM2; ++i)
            for (unsigned int j=0; j<NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }
   }

   // smatrix_sym is always first (skip check test)
   fX2sum = x2sum;
   fC2sum = c2sum;
   if (x2sum == 0 || c2sum == 0) {
      std::cout << "WARNING: x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;
   }

   return 0;
}



// ROOT test


template <unsigned int NDIM1, unsigned int NDIM2>
int TestRunner<NDIM1,NDIM2>::test_tmatrix_kalman() {

   gReporter.Set("TMatrix",NDIM1,NDIM2);


   typedef TMatrixD MnMatrix;
   typedef TVectorD MnVector;

   //   typedef boost::numeric::ublas::matrix<double>  MnMatrix;
   //typedef HepSymMatrix MnSymMatrixHep;


   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "****************************************************************************\n";
   std::cout << "\t\tTMatrix Kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "****************************************************************************\n";

   int npass = NITER;
   TRandom3 r(111);
   double x2sum = 0,c2sum = 0;

   for (int k = 0; k < npass; k++)
   {

      MnMatrix H(first,second);
      MnMatrix K0(second,first);
      MnMatrix Cp(second,second);
      MnMatrix V(first,first);
      MnVector m(first);
      MnVector xp(second);

      // fill matrices with random data
      fillRandomMat(r,H,first,second);
      fillRandomMat(r,K0,second,first);
      fillRandomSym(r,Cp,second);
      fillRandomSym(r,V,first);
      fillRandomVec(r,m,first);
      fillRandomVec(r,xp,second);


      MnMatrix I(second,second);//Identity matrix
      for (int i = 0; i < second; i++)
         for(int j = 0; j <second; j++) {
            I(i,j) = 0.0;
            if (i==j) I(i,i) = 1.0;
         }

      //       if (k==0) {
      //    std::cout << " Cp " << std::endl;
      //    Cp.Print();
      //       }

      {
         double x2 = 0,c2 = 0;
         TVectorD x(second);
         TMatrixD Rtmp(first,first);
         TMatrixD Rinv(first,first);
         TMatrixDSym RinvSym;
         TMatrixD K(second,first);
         TMatrixD C(second,second);
         TMatrixD Ctmp(second,second);
         TVectorD tmp1(first);
         TMatrixD tmp2(second,first);

         TestTimer t(gReporter);
         for (Int_t l = 0; l < NLOOP; l++)
         {
            tmp1 = m; Add(tmp1,-1.0,H,xp);
            x = xp; Add(x,+1.0,K0,tmp1);
            tmp2.MultT(Cp,H);
            Rtmp.Mult(H,tmp2);
            Rinv.Plus(V,Rtmp);
            RinvSym.Use(first,Rinv.GetMatrixArray());
            RinvSym.InvertFast();
            K.Mult(tmp2,Rinv);
            Ctmp.MultT(K,tmp2);
            C.Minus(Cp,Ctmp);
            x2 = RinvSym.Similarity(tmp1);

#ifdef DEBUG
            if (l == 0) {
               std::cout << " Rinv =\n "; Rinv.Print();
               std::cout << " RinvSym =\n "; RinvSym.Print();
               std::cout << " C =\n "; C.Print();
            }
#endif

         }
         x2sum += x2;
         c2 = 0;
         for (unsigned int i=0; i<NDIM2; ++i)
            for (unsigned int j=0; j<NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }

      //   }
   }
   //tr.dump();
   //std::cout << "x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;

   //gReporter.print();

   int iret = 0;
   double d = std::abs(x2sum-fX2sum);
   if ( d > 1.E-6 * fX2sum  ) {
      std::cout << "ERROR: difference found in x2sum = " << x2sum << "\tref = " << fX2sum <<
      "\tdiff = " << d <<  std::endl;
      iret = 1;
   }
   d = std::abs(c2sum-fC2sum);
   if ( d > 1.E-6 * fC2sum  ) {
      std::cout << "ERROR: difference found in c2sum = " << c2sum << "\tref = " << fC2sum <<
      "\tdiff = " << d <<  std::endl;
      iret = 1;
   }


   return iret;
}


// test CLHEP Kalman

#ifdef HAVE_CLHEP
template <unsigned int NDIM1, unsigned int NDIM2>
int TestRunner<NDIM1,NDIM2>::test_clhep_kalman() {


   gReporter.Set("HepMatrix",NDIM1,NDIM2);

   typedef HepSymMatrix MnSymMatrix;
   typedef HepMatrix MnMatrix;
   typedef HepVector MnVector;


   //   typedef boost::numeric::ublas::matrix<double>  MnMatrix;
   //typedef HepSymMatrix MnSymMatrixHep;


   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "****************************************************************************\n";
   std::cout << "  CLHEP Kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "****************************************************************************\n";

   int npass = NITER;
   TRandom3 r(111);
   double x2sum = 0,c2sum = 0;

   for (int k = 0; k < npass; k++)
   {

      // in CLHEP index starts from 1
      MnMatrix H(first,second);
      MnMatrix K0(second,first);
      MnMatrix Cp(second,second);
      MnMatrix V(first,first);
      MnVector m(first);
      MnVector xp(second);

      // fill matrices with random data
      fillRandomMat(r,H,first,second,1);
      fillRandomMat(r,K0,second,first,1);
      fillRandomSym(r,Cp,second,1);
      fillRandomSym(r,V,first,1);
      fillRandomVec(r,m,first);
      fillRandomVec(r,xp,second);

      MnSymMatrix I(second,1);//Identity matrix

      {
         double x2 = 0,c2 = 0;
         MnVector x(second);
         MnMatrix Rinv(first,first);
         MnSymMatrix RinvSym(first);
         MnMatrix K(second,first);
         MnSymMatrix C(second);
         MnVector vtmp1(first);
         MnMatrix tmp(second,first);

         TestTimer t(gReporter);
         int ifail;
         for (Int_t l = 0; l < NLOOP; l++)
         {


            vtmp1 = H*xp -m;
            //x = xp + K0 * (m- H * xp);
            x = xp - K0 * vtmp1;
            tmp = Cp * H.T();
            Rinv = V;  Rinv +=  H * tmp;
            RinvSym.assign(Rinv);
            RinvSym.invert(ifail);
            if (ifail !=0) { std::cout << "Error inverting Rinv" << std::endl; break; }
            K = tmp*RinvSym;
            //C.assign( (I-K*H)*Cp);
            //C = (I-K*H)*Cp;
            C.assign( (I-K*H)*Cp );
            x2= RinvSym.similarity(vtmp1);
            if(ifail!=0) { std::cout << "Error inverting Rinv" << std::endl; break; }
         }
         x2sum += x2;

         c2 = 0;
         for (unsigned int i=1; i<=NDIM2; ++i)
            for (unsigned int j=1; j<=NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }

      //   }
   }
   //tr.dump();
   //std::cout << "x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;

   int iret = 0;
   double d = std::abs(x2sum-fX2sum);
   if ( d > 1.E-6 * fX2sum  ) {
      std::cout << "ERROR: difference found in x2sum = " << x2sum << "\tref = " << fX2sum <<
      "\tdiff = " << d <<  std::endl;
      iret = 1;
   }
   d = std::abs(c2sum-fC2sum);
   if ( d > 1.E-6 * fC2sum  ) {
      std::cout << "ERROR: difference found in c2sum = " << c2sum << "\tref = " << fC2sum <<
      "\tdiff = " << d <<  std::endl;
      iret = 1;
   }

   return iret;
}
#endif


int main(int argc, char *argv[]) {


   if (runTest() ) {
      std::cout << "\nERROR - stressKalman FAILED - exit!" << std::endl ;
      return -1;
   };

   gReporter.print(std::cout);
   std::string fname = "kalman";
   if (argc > 1) {
      std::string platf(argv[1]);
      fname = fname + "_" + platf;
   }

   gReporter.save(fname+".root");

   return 0;
}
