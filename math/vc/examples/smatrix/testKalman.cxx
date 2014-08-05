#ifdef USE_VC
//using namespace Vc;
#include "Vc/Vc"
#endif



#include <cassert>



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


#include <sys/times.h>
#include <unistd.h>

double cpuTime()
{
   struct tms usage;
   times(&usage);
   return ((double) usage.tms_utime) / sysconf(_SC_CLK_TCK);
}

double clockTime()
{
   struct tms usage;
   return ((double) times(&usage)) / sysconf(_SC_CLK_TCK);
}


#ifndef NDIM1
#define NDIM1 5
#endif
#ifndef NDIM2
#define NDIM2 5
#endif

#define NITER 1  // number of iterations

#define NLOOP 500000 // number of time the test is repeted
#define NLISTSIZE 64  // size of matrix/vector lists

using namespace ROOT::Math;

#include "TestTimer.h"

#ifdef USE_VC
typedef Vc::double_v Stype;
#else
typedef double Stype;
#endif

#ifdef USE_VC
const int NLIST = NLISTSIZE / Vc::double_v::Size;
#else
const int NLIST = NLISTSIZE;
#endif


int test_smatrix_kalman() {

   // need to write explicitly the dimensions


   typedef SMatrix<Stype, NDIM1, NDIM1>  MnMatrixNN;
   typedef SMatrix<Stype, NDIM2, NDIM2>  MnMatrixMM;
   typedef SMatrix<Stype, NDIM1, NDIM2>  MnMatrixNM;
   typedef SMatrix<Stype, NDIM2 , NDIM1> MnMatrixMN;
   typedef SMatrix<Stype, NDIM1 >        MnSymMatrixNN;
   typedef SMatrix<Stype, NDIM2 >        MnSymMatrixMM;
   typedef SVector<Stype, NDIM1>         MnVectorN;
   typedef SVector<Stype, NDIM2>         MnVectorM;



   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  SMatrix kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";




   int npass = NITER;
   TRandom3 r(111);
   Stype x2sum = 0.0;
   Stype c2sum = 0.0;

   for (int ipass = 0; ipass < npass; ipass++) {


      MnMatrixNM H[NLIST];
      MnMatrixMN K0[NLIST];
      MnSymMatrixMM Cp[NLIST];
      MnSymMatrixNN V[NLIST];
      MnVectorN m[NLIST];
      MnVectorM xp[NLIST];


      // fill matrices with random data
      for (int j = 0; j < NLIST; j++)  fillRandomMat(r,H[j],first,second);
      for (int j = 0; j < NLIST; j++)  fillRandomMat(r,K0[j],second,first);
      for (int j = 0; j < NLIST; j++)  fillRandomSym(r,Cp[j],second);
      for (int j = 0; j < NLIST; j++)  fillRandomSym(r,V[j],first);
      for (int j = 0; j < NLIST; j++)  fillRandomVec(r,m[j],first);
      for (int j = 0; j < NLIST; j++)  fillRandomVec(r,xp[j],second);


      // MnSymMatrixMM I;
      // for(int i = 0; i < second; i++)
      //   I(i,i) = 1;

#ifdef DEBUG
      std::cout << "pass " << ipass << std::endl;
      if (k == 0) {
         std::cout << " K0 = " << K0[0] << std::endl;
         std::cout << " H = " << K0[0] << std::endl;
         std::cout << " Cp = " << Cp[0] << std::endl;
         std::cout << " V = " << V[0] << std::endl;
         std::cout << " m = " << m[0] << std::endl;
         std::cout << " xp = " << xp[0] << std::endl;
      }
#endif


      {
         Stype x2 = 0.0,c2 = 0.0;
         test::Timer t("SMatrix Kalman ");

         MnVectorM x;
         MnMatrixMN tmp;
         MnSymMatrixNN Rinv;
         MnMatrixMN K;
         MnSymMatrixMM C;
         MnVectorN vtmp1;
         MnVectorN vtmp;

         for (int l = 0; l < NLOOP; l++) {

            // loop on the list of matrices
            for (int k = 0; k < NLIST; k++)  {



               vtmp1 = H[k]*xp[k] -m[k];
               //x = xp + K0 * (m- H * xp);
               x = xp[k] - K0[k] * vtmp1;
               tmp = Cp[k] * Transpose(H[k]);
               Rinv = V[k];  Rinv +=  H[k] * tmp;

               bool test = Rinv.InvertFast();
               if(!test) {
                  std::cout<<"inversion failed" <<std::endl;
                  std::cout << Rinv << std::endl;
               }

               K =  tmp * Rinv ;
               C = Cp[k]; C -= K * Transpose(tmp);
               //C = ( I - K * H ) * Cp;
               //x2 = Product(Rinv,m-H*xp);  // this does not compile on WIN32
               vtmp = m[k]-H[k]*xp[k];
               x2 = Dot(vtmp, Rinv*vtmp);


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
      }

      std::cerr << "SMatrix:    x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;
#ifdef USE_VC
      double sx2=0; double sc2=0;
      for (int l=0;l<Stype::Size; ++l) {
         sx2 += x2sum[l];
         sc2 += c2sum[l];
      }
      std::cerr << "SMatrix:    x2sum = " << sx2 << "\tc2sum = " << sc2 << std::endl;
#endif
   }
   return 0;

}

int test_smatrix_sym_kalman() {

#ifdef TEST_SYM

   // need to write explicitly the dimensions


   typedef SMatrix<Stype, NDIM1, NDIM1>  MnMatrixNN;
   typedef SMatrix<Stype, NDIM2, NDIM2>  MnMatrixMM;
   typedef SMatrix<Stype, NDIM1, NDIM2>  MnMatrixNM;
   typedef SMatrix<Stype, NDIM2 , NDIM1> MnMatrixMN;
   typedef SMatrix<Stype, NDIM1, NDIM1, MatRepSym<Stype, NDIM1> >        MnSymMatrixNN;
   typedef SMatrix<Stype, NDIM2, NDIM2, MatRepSym<Stype, NDIM2> >        MnSymMatrixMM;
   typedef SVector<Stype, NDIM1>         MnVectorN;
   typedef SVector<Stype, NDIM2>         MnVectorM;
   typedef SVector<Stype, NDIM1*(NDIM1+1)/2>   MnVectorN2;
   typedef SVector<Stype, NDIM2*(NDIM2+1)/2>   MnVectorM2;



   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  SMatrix_SyM kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";



   int npass = NITER;
   TRandom3 r(111);
   Stype x2sum = 0.0, c2sum = 0.0;

   for (int ipass = 0; ipass < npass; ipass++) {


      MnMatrixNM H[NLIST];
      MnMatrixMN K0[NLIST];
      MnSymMatrixMM Cp[NLIST];
      MnSymMatrixNN V[NLIST];
      MnVectorN m[NLIST];
      MnVectorM xp[NLIST];


      // fill matrices with random data
      for (int j = 0; j < NLIST; j++)  fillRandomMat(r,H[j],first,second);
      for (int j = 0; j < NLIST; j++)  fillRandomMat(r,K0[j],second,first);
      for (int j = 0; j < NLIST; j++)  fillRandomSym(r,Cp[j],second);
      for (int j = 0; j < NLIST; j++)  fillRandomSym(r,V[j],first);
      for (int j = 0; j < NLIST; j++)  fillRandomVec(r,m[j],first);
      for (int j = 0; j < NLIST; j++)  fillRandomVec(r,xp[j],second);



      // MnSymMatrixMM I;
      // for(int i = 0; i < second; i++)
      //   I(i,i) = 1;

#ifdef DEBUG
      std::cout << "pass " << ipass << std::endl;
      if (ipass == 0) {
         std::cout << " K0 = " << K0 << std::endl;
         std::cout << " H = " << K0 << std::endl;
         std::cout << " Cp = " << Cp << std::endl;
         std::cout << " V = " << V << std::endl;
         std::cout << " m = " << m << std::endl;
         std::cout << " xp = " << xp << std::endl;
      }
#endif


      //double clc1 = clockTime();

      Stype x2 = 0.0,c2 = 0.0;
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

      for (int l = 0; l < NLOOP; l++)  {
         for (int k = 0; k < NLIST; k++) {


#ifdef OPTIMIZED_SMATRIX_SYM
            vtmp1 = H[k]*xp[k] -m[k];
            //x = xp + K0 * (m- H * xp);
            x = xp[k] - K0[k] * vtmp1;
            tmp = Cp[k] * Transpose(H[k]);
            // we are sure that H*tmp result is symmetric
            AssignSym::Evaluate(RinvSym,H[k]*tmp);
            RinvSym += V[k];

            bool test = RinvSym.InvertFast();
            if(!test) {
               std::cout<<"inversion failed" <<std::endl;
               std::cout << RinvSym << std::endl;
            }

            K =  tmp * RinvSym ;
            // we profit from the fact that result of K*tmpT is symmetric
            AssignSym::Evaluate(Ctmp, K*Transpose(tmp) );
            C = Cp[k]; C -= Ctmp;
            //C = ( I - K * H ) * Cp;
            //x2 = Product(Rinv,m-H*xp);  // this does not compile on WIN32
            vtmp = m[k]-H[k]*xp[k];
            x2 = Dot(vtmp, RinvSym*vtmp);
#else
            // use similarity function
            vtmp1 = H[k]*xp[k] -m[k];
            x = xp[k] - K0[k] * vtmp1;
            RinvSym = V[k];  RinvSym +=  Similarity(H[k],Cp[k]);

            bool test = RinvSym.InvertFast();
            if(!test) {
               std::cout<<"inversion failed" <<std::endl;
               std::cout << RinvSym << std::endl;
            }

            Ctmp = SimilarityT(H[k], RinvSym);
            C = Cp[k]; C -= Similarity(Cp[k], Ctmp);
            vtmp = m[k]-H[k]*xp[k];
            x2 = Similarity(vtmp, RinvSym);
#endif

            x2sum += x2;
            c2 = 0;
            for (int i=0; i<NDIM2; ++i)
               for (int j=0; j<NDIM2; ++j)
                  c2 += C(i,j);
            c2sum += c2;

         }  // end loop on list
            //std::cout << k << " chi2 = " << x2 << std::endl;

      }  // end loop on trials

      //tr.dump();

      // double clc2 = clockTime();
      // std::cerr << "Time: " << (clc2 - clc1) << std::endl;

      std::cerr << "SMatrixSym:  x2sum = " << x2sum << "\tc2sum = " << c2sum << std::endl;
#ifdef USE_VC
      double sx2=0; double sc2=0;
      for (int l=0;l<Stype::Size; ++l) {
         sx2 += x2sum[l];
         sc2 += c2sum[l];
      }
      std::cerr << "SMatrixSym:    x2sum = " << sx2 << "\tc2sum = " << sc2 << std::endl;
#endif


#endif
   }

   return 0;

}



// ROOT test


int test_tmatrix_kalman() {

#ifdef USE_TMATRIX



   typedef TMatrixD MnMatrix;
   typedef TVectorD MnVector;

   //   typedef boost::numeric::ublas::matrix<Stype>  MnMatrix;
   //typedef HepSymMatrix MnSymMatrixHep;


   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  TMatrix Kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";



   int npass = NITER;
   TRandom3 r(111);
   gMatrixCheck = 0;
   Stype x2sum = 0,c2sum = 0;

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
         Stype x2 = 0,c2 = 0;
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

#endif
   return 0;

}



// test CLHEP Kalman

#ifdef HAVE_CLHEP
int test_clhep_kalman() {



   typedef HepSymMatrix MnSymMatrix;
   typedef HepMatrix MnMatrix;
   typedef HepVector MnVector;


   //   typedef boost::numeric::ublas::matrix<Stype>  MnMatrix;
   //typedef HepSymMatrix MnSymMatrixHep;


   int first = NDIM1;  //Can change the size of the matrices
   int second = NDIM2;


   std::cout << "************************************************\n";
   std::cout << "  CLHEP Kalman test  "   <<  first << " x " << second  << std::endl;
   std::cout << "************************************************\n";



   int npass = NITER;
   TRandom3 r(111);
   Stype x2sum = 0,c2sum = 0;

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
         Stype x2 = 0,c2 = 0;
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

   //int nlist = NLISTSIZE;
#ifdef USE_VC
   std::cout << "Using VC library - size = " << ROOT::Vc::double_v::Size << " VC_IMPL = " << VC_IMPL << std::endl;
   //nlist /= Vc::double_v::Size;
#endif

   std::cout << " making vector/matrix lists of size = " << NLIST << std::endl;


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
