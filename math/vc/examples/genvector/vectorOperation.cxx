// test performance of all vectors operations +,- and *



#include <cassert>

#ifdef USE_VDT
#include "vdtMath.h"
#endif

//#define USE_VC
#ifdef USE_VC
#include "Vc/Vc"
#include <Vc/Allocator>
typedef Vc::double_v Double_type;
#define ZERO Vc::Zero
#else
typedef double Double_type;
#define ZERO 0
#endif


#include "TRandom2.h"
#include "TStopwatch.h"

#include <vector>
#include <iostream>
#include <iomanip>

#ifdef DIM_2
#ifdef USE_POINT
#include "Math/Point2D.h"
typedef ROOT::Math::XYPoint VecType;
#elif USE_TVECTOR
#include "TVector2.h"
typedef TVector2 VecType;
#else
#include "Math/Vector2D.h"
typedef ROOT::Math::XYVector VecType;
#endif

#ifndef USE_ROOT
#define VSUM(v) v.x() + v.y()
#else
#define VSUM(v) v.X() + v.Y()
#endif


#elif DIM_3 // 3 Dimensions

#ifdef USE_POINT
#include "Math/Point3D.h"
typedef ROOT::Math::XYZPoint VecType;
#elif USE_TVECTOR
#include "TVector3.h"
typedef TVector3 VecType;
#else
#include "Math/Vector3D.h"
typedef ROOT::Math::XYZVector VecType;
#endif

#ifndef USE_TVECTOR
#define VSUM(v) v.x() + v.y() + v.z()
#else
#define VSUM(v) v.X() + v.Y() + v.Z()
#endif

#else // default is 4D

#undef USE_POINT
#if USE_TVECTOR
#include "TLorentzVector.h"
typedef TLorentzVector VecType;
#else
#include "Math/Vector4D.h"
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double_type> > VecType;
//#ifdef USE_VC
//Vc_DECLARE_ALLOCATOR(VecType)
//#endif
#endif

#include "Math/VectorUtil.h"

#ifndef USE_TVECTOR
#define VSUM(v) v.x() + v.y() + v.z()
//#define VSUM(v) v.x()
#else
#define VSUM(v) v.X() + v.Y() + v.Z()
#endif


#endif

//#define VLISTSIZE 8
#define VLISTSIZE 100

#ifdef USE_VC
const int N = VLISTSIZE/Vc::double_v::Size;
#else
const int N = VLISTSIZE;
#endif

const int NLOOP = 5000000;
//const int NLOOP = 1;

const int NOP = NLOOP*VLISTSIZE;
const double TSCALE = 1.E9/double(NOP);

template<class Vector>
class TestVector {
public:

   TestVector();

   void Operations();

   void Add();
   void Add2();
   void Sub();
   void Sub2();
   void Scale();
   void Scale2();
   void Divide();
   void Divide2();

   void MathFunction_sin();
   void MathFunction_exp();
   void MathFunction_log();
   void MathFunction_atan();
   void MathFunction_atan2();

   void Boost();

   void Read();

   void PrintSummary();

   void PrintResult(Double_type s);

private:

   std::vector<Vector> vlist;
   std::vector<Vector> vlist2;
   std::vector<Double_type> scale;
   std::vector <std::vector <double > >  vcoords;
   double fTime[50]; // timing results
   int  fTest;
};

//utility function to aggregate vectors in a vc type
template <class Vector, class Vector_V, int Size>
void MakeVcVector( const Vector * vlist, Vector_V & vret ) {
   const int dim = 4;
   typename Vector_V::Scalar vcoord[dim];
   typename Vector::Scalar coord[dim];
   for (int i = 0; i < Size; ++i) {
      vlist[i].GetCoordinates(coord);
      for (int j = 0; j < dim; ++j)
         vcoord[j][i] = coord[j];
   }
   vret.SetCootdinates(vcoord);
   return vret;
}
// template <class Vector, class Vector_V, int Size>
// void UnpackVcVector( const Vector * vlist, Vector_V & vret ) {
//    const int dim = 4;
//    typename Vector_V::Scalar vcoord[dim];
//    typename Vector::Scalar coord[dim];
//    for (int i = 0; i < Size; ++i) {
//       vlist[i].GetCoordinates(coord);
//       for (int j = 0; j < dim; ++j)
//          vcoord[j][i] = coord[j];
//    }
//    vret.SetCootdinates(vcoord);
//    return vret;
// }


template<class Vector>
TestVector<Vector>::TestVector() :
   vlist(N),
   vlist2(N),
   scale(N),
#ifdef USE_VC
   vcoords(N*Vc::double_v::Size),
#else
   vcoords(N),
#endif
   fTest(0)
{
   // create list of vectors and fill them

   TRandom2 r(111);

   double coord[4];
   for (int i = 0; i< N; ++i) {
#ifndef USE_VC
      Double_type x = r.Uniform(-1,1);
      Double_type y = r.Uniform(-1,1);
      Double_type z = r.Uniform(-1,1);
      Double_type t = r.Uniform(2,10);
      Double_type s = r.Uniform(0,1);
      coord[0] = x; coord[1] = y; coord[2] = z; coord[3] = t;
      vcoords[i] = std::vector<double>(coord,coord+4);
#else
      Double_type x = 0.;
      Double_type y = 0.;
      Double_type z = 0.;
      Double_type t = 0.;
      Double_type s = 0.;
      for (int j = 0; j< Vc::double_v::Size; ++j) {
         x[j] = r.Uniform(-1,1);
         y[j] = r.Uniform(-1,1);
         z[j] = r.Uniform(-1,1);
         t[j] = r.Uniform(2,10);
         s[j] = r.Uniform(0,1);
         coord[0] = x[j]; coord[1] = y[j]; coord[2] = z[j]; coord[3] = t[j];
         vcoords[i*Vc::double_v::Size+j]  = std::vector<double>(coord,coord+4);
      }
#endif

#ifdef DIM_2
      vlist[i] = Vector( x, y );
#elif DIM_3
      vlist[i] = Vector( x, y, z);
#else // 4D
      vlist[i] = Vector( x, y, z, t);
#endif
      scale[i] = s;
   }

   std::cout << "test using " << typeid(vlist[0]).name() << std::endl;
   std::cout << "Vector used " << vlist[0] << std::endl;
   std::cout << "Vector used " << vlist[1] << std::endl;

   // create second list of vectors which is same vector shifted by 1
   for (int i = 0; i< N; ++i) {
#ifndef USE_VC
      vlist2[i] = (i < N-1) ? vlist[i+1] : vlist[0];
#else
      Double_type x1 = vlist[i].X();
      Double_type y1 = vlist[i].Y();
      Double_type z1 = vlist[i].Z();
      Double_type t1 = vlist[i].E();
      Double_type x2 = (i< N-1) ? vlist[i+1].X() : vlist[0].X();
      Double_type y2 = (i< N-1) ? vlist[i+1].Y() : vlist[0].Y();
      Double_type z2 = (i< N-1) ? vlist[i+1].Z() : vlist[0].Z();
      Double_type t2 = (i< N-1) ? vlist[i+1].E() : vlist[0].E();
      Double_type x;
      Double_type y;
      Double_type z;
      Double_type t;
      int j = 0;
      for (j = 0; j< Vc::double_v::Size-1; ++j) {
         x[j] = x1[j+1];
         y[j] = y1[j+1];
         z[j] = z1[j+1];
         t[j] = t1[j+1];
      }
      j = Vc::double_v::Size-1;
      x[j] = x2[0];
      y[j] = y2[0];
      z[j] = z2[0];
      t[j] = t2[0];
      vlist2[i] =  Vector( x, y, z, t);
#endif
   }

}


template<class Vector>
void TestVector<Vector>::PrintResult(Double_type s)
   // print result
{
#ifndef USE_VC
   std::cout << "value " << s << std::endl << std::endl;
#else
   Double_t s2 = 0;
   for (int i = 0; i < Vc::double_v::Size; ++i)
      s2 += s[i];
//   std::cout << "s = " << s << " sum ";
   std::cout << "value " << s2 << std::endl << std::endl;
#endif
}

template<class Vector>
void TestVector<Vector>::Read()
   // just read vector
{
   TStopwatch w;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      //Vector v0 = vlist[l%N];
      //s += v0.X();
      //s = 0;
      for (int i = 0; i< N; ++i) {
         Vector v3 = vlist[i];
         s += VSUM(v3);
         // if (l == 0) {
         //    std::cout << v3 << " sum   " << VSUM(v3) << " total sum " << s << std::endl;
         // }
      }
   }

   std::cout << "Time for  read v3 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime();
}


template<class Vector>
void TestVector<Vector>::Add()
   // normal addition
{
   TStopwatch w;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         Vector v3 = vlist[i] + vlist2[i];
         s += VSUM(v3);
//         std::cout << vlist[i] << "  " << vlist2[i] << std::endl;
         //s += v3.X();
//         PrintResult(s);
      }
   }

   std::cout << "Time for  v3 = v1 + v2 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

template<class Vector>
void TestVector<Vector>::Add2()
{
   // self addition
   TStopwatch w;
   Vector v3;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      v3 = Vector();
      for (int i = 0; i< N; ++i) {
         v3 += vlist[i];
      }
      // if I put s inside inner loop break autivec
      // but also Vc cannot work (will compute something different)
      //s += v3.X();  (if I use this compiler can auto-vectorize)
      s += VSUM(v3);
   }

   std::cout << "Time for  v2 += v1     :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

template<class Vector>
void TestVector<Vector>::Sub()
{
   // normal sub
   TStopwatch w;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         Vector v3 = vlist[i] - vlist2[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v3 = v1 - v2 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

template<class Vector>
void TestVector<Vector>::Sub2()
{
   // self subtruction
   TStopwatch w;
   Vector v3;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 -= vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 -= v1     :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}


template<class Vector>
void TestVector<Vector>::Scale()
{
// normal multiply
   TStopwatch w;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         Vector v3 = scale[i]*vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 = A * v1 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

template<class Vector>
void TestVector<Vector>::Scale2()
{
   // self scale
   TStopwatch w;
   Vector v3;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 = vlist[i];
         v3 *= scale[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v *= a     :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

template<class Vector>
void TestVector<Vector>::Divide()
{
// normal divide
   TStopwatch w;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         Vector v3 = vlist[i]/scale[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 = v1 / a :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

template<class Vector>
void TestVector<Vector>::Divide2()
{
   // self divide
   TStopwatch w;
   Vector v3;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 = vlist[i];
         v3 /= scale[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v /= a      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

#ifdef CASE1
template<class Vector>
void TestVector<Vector>::Operations()
{
   // test several operations
   TStopwatch w;
   Vector v1;
   Vector v2;
   Vector v3;
   Vector v4;
   w.Start();
   Double_type s(0.0);
   const int n = sqrt(N)+0.5;
   for (int l = 0; l<NLOOP; ++l) {
#ifndef USE_VC
      for (int i = 0; i< n; ++i) {
         for (int j = 0; j< i; ++j) {
            s +=  ROOT::Math::VectorUtil::InvariantMass(vlist[i], vlist[j] );
            //std::cout << "inv mass of " << vlist[i] << "  " << vlist[j] << " is " << s << std::endl;
         }
         //v2 = vlist[i]*scale[i];
         // v1 *= scale[i-1];
         // v2 /= scale[i];
         // v3 = v1 + v2;
         // v4 = v1 - v2;
         //s += ROOT::Math::VectorUtil::InvariantMass2(v3,v4);
         //s += ROOT::Math::VectorUtil::InvariantMass2(v1,v2);
         //s += std::sin(v1.E() );
      }
#else
      const int nn = sqrt(vcoords.size()) + 0.5;
      int ncomb = nn*(nn-1)/2;
      std::vector<double> vc1x(ncomb+4);
      std::vector<double> vc2x(ncomb+4);
      std::vector<double> vc1y(ncomb+4);
      std::vector<double> vc2y(ncomb+4);
      std::vector<double> vc1z(ncomb+4);
      std::vector<double> vc2z(ncomb+4);
      std::vector<double> vc1t(ncomb+4);
      std::vector<double> vc2t(ncomb+4);
      double c1[4];
      double c2[4];
      int k = 0;
      for (int i = 0; i< nn; ++i) {
         std::copy(vcoords[i].begin(), vcoords[i].end(),c1);
         //std::cout << "vcoord " << vcoords[i][0] << "  " << c1[0] << std::endl;
         for (int j = 0; j< i; ++j) {
            std::copy(vcoords[j].begin(), vcoords[j].end(),c2);
            vc1x[k] = c1[0];
            vc2x[k] = c2[0];
            vc1y[k] = c1[1];
            vc2y[k] = c2[1];
            vc1z[k] = c1[2];
            vc2z[k] = c2[2];
            vc1t[k] = c1[3];
            vc2t[k] = c2[3];
            k++;
         }
      }
      int ncomb2 = ncomb/Vc::double_v::Size;
      if (ncomb%Vc::Double_v::Size != 0) ncomb2 += 1;
      Vector v1; Vector v2;
      for (int i = 0; i< ncomb2; ++i) {

         typename Vector::Scalar cv[4];
         cv[0].load( &vc1x[i*Vc::double_v::Size], Vc::Unaligned );
         cv[1].load( &vc1y[i*Vc::double_v::Size], Vc::Unaligned );
         cv[2].load( &vc1z[i*Vc::double_v::Size], Vc::Unaligned );
         cv[3].load( &vc1t[i*Vc::double_v::Size], Vc::Unaligned );
         //std::cout << cv[0] << "  " << vc1x[i*Vc::double_v::Size] << std::endl;
         v1.SetCoordinates( cv);

         typename Vector::Scalar cv2[4];
         cv2[0].load( &vc2x[i*Vc::double_v::Size], Vc::Unaligned );
         cv2[1].load( &vc2y[i*Vc::double_v::Size], Vc::Unaligned );
         cv2[2].load( &vc2z[i*Vc::double_v::Size], Vc::Unaligned );
         cv2[3].load( &vc2t[i*Vc::double_v::Size], Vc::Unaligned );
         //std::cout << cv[0] << "  " << vc2x[i*Vc::double_v::Size] << std::endl;
         v2.SetCoordinates( cv2);

         s+= ROOT::Math::VectorUtil::InvariantMass(v1, v2);
         //std::cout << "inv mass of " << v1 << "  " << v2 << " is " << s << std::endl;
      }

#endif
   }

   std::cout << "Time for  Operation      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

#else

template<class Vector>
void TestVector<Vector>::Operations()
{
   // test several operations
   TStopwatch w;
   Vector v1;
   Vector v2;
   Vector v3;
   Vector v4;
   w.Start();
   Double_type s(0.0);

#ifndef USE_VC
   const int n = sqrt(N)+0.5;
#endif

   const int nn = sqrt(vcoords.size()) + 0.5;
   int ncomb = nn*(nn-1)/2;
   std::vector<double> vc1x(ncomb+4);
   std::vector<double> vc2x(ncomb+4);
   std::vector<double> vc1y(ncomb+4);
   std::vector<double> vc2y(ncomb+4);
   std::vector<double> vc1z(ncomb+4);
   std::vector<double> vc2z(ncomb+4);
   std::vector<double> vc1t(ncomb+4);
   std::vector<double> vc2t(ncomb+4);
   double c1[4];
   double c2[4];
   int k = 0;
   for (int i = 0; i< nn; ++i) {
      std::copy(vcoords[i].begin(), vcoords[i].end(),c1);
      //std::cout << "vcoord " << vcoords[i][0] << "  " << c1[0] << std::endl;
      for (int j = 0; j< i; ++j) {
         std::copy(vcoords[j].begin(), vcoords[j].end(),c2);
         vc1x[k] = c1[0];
         vc2x[k] = c2[0];
         vc1y[k] = c1[1];
         vc2y[k] = c2[1];
         vc1z[k] = c1[2];
         vc2z[k] = c2[2];
         vc1t[k] = c1[3];
         vc2t[k] = c2[3];
         k++;
      }
   }



   for (int l = 0; l<NLOOP; ++l) {
#ifndef USE_VC
      for (int i = 0; i< n; ++i) {
         for (int j = 0; j< i; ++j) {
            s +=  ROOT::Math::VectorUtil::InvariantMass(vlist[i], vlist[j] );
            //std::cout << "inv mass of " << vlist[i] << "  " << vlist[j] << " is " << s << std::endl;
         }
         //v2 = vlist[i]*scale[i];
         // v1 *= scale[i-1];
         // v2 /= scale[i];
         // v3 = v1 + v2;
         // v4 = v1 - v2;
         //s += ROOT::Math::VectorUtil::InvariantMass2(v3,v4);
         //s += ROOT::Math::VectorUtil::InvariantMass2(v1,v2);
         //s += std::sin(v1.E() );
      }
#else
      int ncomb2 = ncomb/Vc::double_v::Size;
      if (ncomb%Vc::double_v::Size != 0) ncomb2 += 1;
      Vector v1; Vector v2;
      for (int i = 0; i< ncomb2; ++i) {

         typename Vector::Scalar cv[4];
         cv[0].load( &vc1x[i*Vc::double_v::Size], Vc::Unaligned );
         cv[1].load( &vc1y[i*Vc::double_v::Size], Vc::Unaligned );
         cv[2].load( &vc1z[i*Vc::double_v::Size], Vc::Unaligned );
         cv[3].load( &vc1t[i*Vc::double_v::Size], Vc::Unaligned );
         //std::cout << cv[0] << "  " << vc1x[i*Vc::double_v::Size] << std::endl;
         v1.SetCoordinates( cv);

         typename Vector::Scalar cv2[4];
         cv2[0].load( &vc2x[i*Vc::double_v::Size], Vc::Unaligned );
         cv2[1].load( &vc2y[i*Vc::double_v::Size], Vc::Unaligned );
         cv2[2].load( &vc2z[i*Vc::double_v::Size], Vc::Unaligned );
         cv2[3].load( &vc2t[i*Vc::double_v::Size], Vc::Unaligned );
         //std::cout << cv[0] << "  " << vc2x[i*Vc::double_v::Size] << std::endl;
         v2.SetCoordinates( cv2);

         s+= ROOT::Math::VectorUtil::InvariantMass(v1, v2);
         // std::cout << s << std::endl;
         // PrintResult(s);
         //std::cout << "inv mass of " << v1 << "  " << v2 << " is " << s << std::endl;
      }

      // std::cout << " \n" << std::endl;
      // std::cout << "End trial " << l << std::endl;
      // PrintResult(s);
      // std::cout << " \n" << std::endl;
#endif
   }

   std::cout << "Time for  Operation      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}
#endif


template<class Vector>
void TestVector<Vector>::Boost()
{
   // test several operations
   TStopwatch w;
   Vector v1;
   Vector v2;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         v1 = vlist[i];
         v2 = ROOT::Math::VectorUtil::boostX(v1,scale[i]);
         // s +=  Vc::atan2(v1.Y(),v1.X())  - Vc::atan2(v2.Y(),v2.X());
         //s += VSUM(v2);
      }
   }

   std::cout << "Time for  Boost      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}



template<class Vector>
void TestVector<Vector>::MathFunction_exp()
{
   // test math function
   TStopwatch w;
   Vector v1;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP/10; ++l) {
      for (int i = 0; i< N; ++i) {
         v1 = vlist[i];
         s += std::exp( v1.X() - v1.Y() );
      }
   }

   std::cout << "Time for Exp Function      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE*10;
}



template<class Vector>
void TestVector<Vector>::MathFunction_log()
{
   // test several operations
   TStopwatch w;
   Vector v1;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP/10; ++l) {
      for (int i = 0; i< N; ++i) {
         v1 = vlist[i];
         s += std::log( std::abs(v1.X() ) );
      }
   }

   std::cout << "Time for Log Function      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE*10;
}

template<class Vector>
void TestVector<Vector>::MathFunction_sin()
{
   // test math function
   TStopwatch w;
   Vector v1;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP/100; ++l) {
      for (int i = 0; i< N; ++i) {
         v1 = vlist[i];
         s += std::sin( v1.X() );
      }
   }

   std::cout << "Time for Sin Function      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE*100;
}

template<class Vector>
void TestVector<Vector>::MathFunction_atan()
{
   // test several operations
   TStopwatch w;
   Vector v1;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP/100; ++l) {
      for (int i = 0; i< N; ++i) {
         v1 = vlist[i]; ///scale[i];
         s += std::atan( v1.Y()/v1.X() );
      }
   }

   std::cout << "Time for Atan Function      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE*100;
}

template<class Vector>
void TestVector<Vector>::MathFunction_atan2()
{
   // test several operations
   TStopwatch w;
   Vector v1;
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP/100; ++l) {
      for (int i = 0; i< N; ++i) {
         v1 = vlist[i]; ///scale[i];
         s += std::atan2( v1.Y(),v1.X() );
      }
   }

   std::cout << "Time for Atan2 Function      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE*100;
}

#ifdef HAS_VDT // use VDT
template<class Vector>
void TestVector<Vector>::MathFunction()
{
   // test several operations
   TStopwatch w;
   Vector v;
   double x[N];
   double r[N];
   w.Start();
   Double_type s(0.0);
   for (int l = 0; l<NLOOP; ++l) {
      for (int i = 0; i< N; ++i) {
         Vector v = vlist[i];
         //x[i] = v.X()/v.Pt() ;
         x[i] = v.X()/v.Y();
      }
      vdt::fast_atanv(N,x,r);
      for (int i = 0; i< N; ++i) {
         s += x[i];
      }

      //s += sin(v1.E() );
   }

   std::cout << "Time for MathFUnction(VDT)      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   PrintResult(s);
   fTime[fTest++] = w.CpuTime()*TSCALE;
}

#endif

template<class Vector>
void TestVector<Vector>::PrintSummary()
{
   std::cout << "\nResults for " << typeid(vlist[0]).name() << std::endl;
   std::cout << " v3 = v1+v2"
             << " v2 += v1  "
             << " v3 = v1-v2"
             << " v2 -= v1  "
             << " v2 = a*v1 "
             << " v1 *= a   "
             << " v2 = v1/a "
             << " v1 /= a   "
             << " log       "
             << " exp       "
             << " sin       "
             << " atan      "
             << " atan2     "
             << std::endl;

   // start from 3
   for (int i = 3; i < fTest; ++i) {
      std::cout << std::setw(8) << fTime[i] << "   ";
   }
   std::cout << std::endl << std::endl;
}

int main() {
   TestVector<VecType> t;

#ifdef USE_VC
   std::cout << "testing using Vc: Vc size is " << Vc::double_v::Size << " Looping on " << N << " vectors" << std::endl;
   std::cout << "Implementation type:  " << VC_IMPL << std::endl;
#else
   std::cout << "testing using standard double's. Looping on " << N << " vectors" << std::endl;
#endif


   t.Read();

   t.Operations();
   t.Boost();


#ifndef USE_POINT
   t.Add();
   t.Add2();
   t.Sub();
   t.Sub2();
#endif
   t.Scale();
   t.Scale2();
#ifndef USE_TVECTOR
   t.Divide();
   t.Divide2();
#endif


   t.MathFunction_log();
   t.MathFunction_exp();
   t.MathFunction_sin();
   t.MathFunction_atan();
   t.MathFunction_atan2();





   // summurize test
   t.PrintSummary();
}
