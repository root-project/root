

#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include "TMatrixD.h"
#include "TVectorD.h"

#include "TRandom3.h"

#include "matrix_util.h"


#define TEST_SYM

//#define HAVE_CLHEP
#ifdef HAVE_CLHEP
#include "CLHEP/Matrix/SymMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#endif

// #include "SealUtil/SealTimer.h"
// #include "SealUtil/SealHRRTChrono.h"
// #include "SealUtil/TimingReport.h"

#include <iostream>

#ifndef NDIM1
#define NDIM1 2
#endif
#ifndef NDIM2
#define NDIM2 5
#endif

#define NITER 1  // number of iterations

#define NLOOP 1000000 // number of time the test is repeted

using namespace ROOT::Math;

#include "TestTimer.h"

int test_smatrix_kalman() {

   // need to write explicitly the dimensions


   typedef SMatrix<double, NDIM1, NDIM1>  MnMatrixNN;
   typedef SMatrix<double, NDIM2, NDIM2>  MnMatrixMM;
   typedef SMatrix<double, NDIM1, NDIM2>  MnMatrixNM;
   typedef SMatrix<double, NDIM2 , NDIM1> MnMatrixMN;
   typedef SMatrix<double, NDIM1 >        MnSymMatrixNN;
   typedef SMatrix<double, NDIM2 >        MnSymMatrixMM;
   typedef SVector<double, NDIM1>         MnVectorN;
   typedef SVector<double, NDIM2>         MnVectorM;



   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  SMatrix kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";




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
         test::Timer t("SMatrix Kalman ");

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

         }
         //std::cout << k << " chi2 = " << x2 << std::endl;
         x2sum += x2;
         c2 = 0;
         for (int i=0; i<NDIM2; ++i)
            for (int j=0; j<NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }
   }
   //tr.dump();

   std::cerr << "SMatrix:    x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;

   return 0;
}

#ifdef TEST_SYM
int test_smatrix_sym_kalman() {

   // need to write explicitly the dimensions


   typedef SMatrix<double, NDIM1, NDIM1>  MnMatrixNN;
   typedef SMatrix<double, NDIM2, NDIM2>  MnMatrixMM;
   typedef SMatrix<double, NDIM1, NDIM2>  MnMatrixNM;
   typedef SMatrix<double, NDIM2 , NDIM1> MnMatrixMN;
   typedef SMatrix<double, NDIM1, NDIM1, MatRepSym<double, NDIM1> >        MnSymMatrixNN;
   typedef SMatrix<double, NDIM2, NDIM2, MatRepSym<double, NDIM2> >        MnSymMatrixMM;
   typedef SVector<double, NDIM1>         MnVectorN;
   typedef SVector<double, NDIM2>         MnVectorM;
   typedef SVector<double, NDIM1*(NDIM1+1)/2>   MnVectorN2;
   typedef SVector<double, NDIM2*(NDIM2+1)/2>   MnVectorM2;



   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  SMatrix_SyM kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";




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
         test::Timer t("SMatrix Kalman ");

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
            AssignSym::Evaluate(RinvSym,H*tmp);
            RinvSym += V;

            bool test = RinvSym.Invert();
            if(!test) {
               std::cout<<"inversion failed" <<std::endl;
               std::cout << RinvSym << std::endl;
            }

            K =  tmp * RinvSym ;
            // we profit from the fact that result of K*tmpT is symmetric
            AssignSym::Evaluate(Ctmp, K*Transpose(tmp) );
            C = Cp; C -= Ctmp;
            //C = ( I - K * H ) * Cp;
            //x2 = Product(Rinv,m-H*xp);  // this does not compile on WIN32
            vtmp = m-H*xp;
            x2 = Dot(vtmp, RinvSym*vtmp);
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

            Ctmp = SimilarityT(H, RinvSym);
            C = Cp; C -= Similarity(Cp, Ctmp);
            vtmp = m-H*xp;
            x2 = Similarity(vtmp, RinvSym);
#endif

         }
         //std::cout << k << " chi2 = " << x2 << std::endl;
         x2sum += x2;
         c2 = 0;
         for (int i=0; i<NDIM2; ++i)
            for (int j=0; j<NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }
   }
   //tr.dump();

   std::cerr << "SMatrixSym:  x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;

   return 0;
}

#endif


// ROOT test


int test_tmatrix_kalman() {




   typedef TMatrixD MnMatrix;
   typedef TVectorD MnVector;

   //   typedef boost::numeric::ublas::matrix<double>  MnMatrix;
   //typedef HepSymMatrix MnSymMatrixHep;


   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  TMatrix Kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";



   int npass = NITER;
   TRandom3 r(111);
   gMatrixCheck = 0;
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
         TMatrixD Rinv(first,first);
         TMatrixD Rtmp(first,first);
         TMatrixDSym RinvSym;
         TMatrixD K(second,first);
         TMatrixD C(second,second);
         TMatrixD Ctmp(second,second);
         TVectorD tmp1(first);
         TMatrixD tmp2(second,first);
#define OPTIMIZED_TMATRIX
#ifndef OPTIMIZED_TMATRIX
         TMatrixD HT(second,first);
         TMatrixD tmp2T(first,second);
#endif

         test::Timer t("TMatrix Kalman ");
         for (Int_t l = 0; l < NLOOP; l++)
         {
#ifdef OPTIMIZED_TMATRIX
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

#else
            tmp1 = H*xp -m;
            //x = xp + K0 * (m- H * xp);
            x = xp - K0 * tmp1;
            tmp2 = Cp * HT.Transpose(H);
            Rinv = V;  Rinv +=  H * tmp2;
            RinvSym.Use(first,Rinv.GetMatrixArray());
            RinvSym.InvertFast();
            K= tmp2* Rinv;
            C = Cp; C -= K*tmp2T.Transpose(tmp2);
            x2= RinvSym.Similarity(tmp1);
#endif

         }
         x2sum += x2;
         c2 = 0;
         for (int i=0; i<NDIM2; ++i)
            for (int j=0; j<NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }

      //   }
   }
   //tr.dump();
   std::cerr << "TMatrix:     x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;

   return 0;
}


// test CLHEP Kalman

#ifdef HAVE_CLHEP
int test_clhep_kalman() {



   typedef HepSymMatrix MnSymMatrix;
   typedef HepMatrix MnMatrix;
   typedef HepVector MnVector;


   //   typedef boost::numeric::ublas::matrix<double>  MnMatrix;
   //typedef HepSymMatrix MnSymMatrixHep;


   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  CLHEP Kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";



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

         test::Timer t("CLHEP Kalman ");
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
         // std::cout << k << " chi2 " << x2 << std::endl;
         x2sum += x2;

         c2 = 0;
         for (int i=1; i<=NDIM2; ++i)
            for (int j=1; j<=NDIM2; ++j)
               c2 += C(i,j);
         c2sum += c2;
      }

      //   }
   }
   //tr.dump();
   std::cerr << "x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;

   return 0;
}
#endif



int testKalman() {

#ifdef TEST_SYM
   test_smatrix_sym_kalman();
#endif

   test_smatrix_kalman();
   test_tmatrix_kalman();
#ifdef HAVE_CLHEP
   test_clhep_kalman();
#endif

   return 0;


}

int main() {
   return testKalman();
}
