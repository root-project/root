// test performance of all vectors operations +,- and *

// results on mactelm g++ 4.01 showing ROOT::Math performs best overall

//v3 = v1+v2 v2 += v1   v3 = v1-v2 v2 -= v1   v2 = a*v1  v1 *= a    v2 = v1/a  v1 /= a
//    0.59       0.57       0.58       0.56       0.69        0.7       1.65       1.64     2D
//    0.79       0.79       0.78        0.8       0.97       0.95       1.85       1.84     3D
//    1.07       1.07       1.07       1.07       1.32       1.31       1.72       1.71     4D

//  ROOT Physics Vector (TVector's):
//v3 = v1+v2 v2 += v1   v3 = v1-v2 v2 -= v1   v2 = a*v1  v1 *= a
//    4.4        0.97       4.41       0.96       4.43       1.13          2D
//    5.44       1.25       5.48       1.24       6.12       1.46          3D
//   17.65       7.32      17.65       7.35      10.25       7.79          4D

//  CLHEP Vector (HepVector's):
//v3 = v1+v2 v2 += v1   v3 = v1-v2 v2 -= v1   v2 = a*v1  v1 *= a
//    0.57       0.55       0.56       0.55        0.7        0.7                           2D
//     0.8       0.79       0.78       0.77       0.96       0.94        2.7        3.7     3D
//    1.06       1.02       1.06       1.02       1.26       1.26       2.99       3.98     4D




#include "TRandom2.h"
#include "TStopwatch.h"

#include <vector>
#include <iostream>
#include <iomanip>

#ifdef DIM_2
#ifdef USE_POINT
#include "Math/Point2D.h"
typedef ROOT::Math::XYPoint VecType;
#elif USE_CLHEP
#include "CLHEP/Vector/TwoVector.h"
typedef Hep2Vector VecType;
#elif USE_ROOT
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
#elif USE_CLHEP
#include "CLHEP/Vector/ThreeVector.h"
typedef Hep3Vector VecType;
#elif USE_ROOT
#include "TVector3.h"
typedef TVector3 VecType;
#else
#include "Math/Vector3D.h"
typedef ROOT::Math::XYZVector VecType;
#endif

#ifndef USE_ROOT
#define VSUM(v) v.x() + v.y() + v.z()
#else
#define VSUM(v) v.X() + v.Y() + v.Z()
#endif

#else // default is 4D

#undef USE_POINT
#ifdef USE_CLHEP
#include "CLHEP/Vector/LorentzVector.h"
typedef HepLorentzVector VecType;
#elif USE_ROOT
#include "TLorentzVector.h"
typedef TLorentzVector VecType;
#else
#include "Math/Vector4D.h"
typedef ROOT::Math::XYZTVector VecType;
#endif

#ifndef USE_ROOT
#define VSUM(v) v.x() + v.y() + v.z() + v.t()
#else
#define VSUM(v) v.X() + v.Y() + v.Z() + v.T()
#endif

#endif


const int N = 1000000;

template<class Vector>
class TestVector {
public:

   TestVector();
   void Add();
   void Add2();
   void Sub();
   void Sub2();
   void Scale();
   void Scale2();
   void Divide();
   void Divide2();

   void PrintSummary();

private:

   std::vector<Vector> vlist;
   std::vector<double> scale;
   double fTime[10]; // timing results
   int  fTest;
};



template<class Vector>
TestVector<Vector>::TestVector() :
   vlist(std::vector<Vector>(N) ),
   scale(std::vector<double>(N) ),
   fTest(0)
{

   // create list of vectors and fill them

   TRandom2 r(111);
   for (int i = 0; i< N; ++i) {
#ifdef DIM_2
      vlist[i] = Vector( r.Uniform(-1,1), r.Uniform(-1,1) );
#elif DIM_3
      vlist[i] = Vector( r.Uniform(-1,1),r.Uniform(-1,1),r.Uniform(-1,1)  );
#else // 4D
      vlist[i] = Vector( r.Uniform(-1,1),r.Uniform(-1,1),r.Uniform(-1,1), r.Uniform(2,10) );
#endif
      scale[i] = r.Uniform(0,1);
   }

   std::cout << "test using " << typeid(vlist[0]).name() << std::endl;
}

template<class Vector>
void TestVector<Vector>::Add()
   // normal addition
{
   TStopwatch w;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 1; i< N; ++i) {
         Vector v3 = vlist[i-1] + vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v3 = v1 + v2 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}

template<class Vector>
void TestVector<Vector>::Add2()
{
   // self addition
   TStopwatch w;
   Vector v3;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 += vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 += v1     :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}

template<class Vector>
void TestVector<Vector>::Sub()
{
   // normal sub
   TStopwatch w;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 1; i< N; ++i) {
         Vector v3 = vlist[i-1] - vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v3 = v1 - v2 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}

template<class Vector>
void TestVector<Vector>::Sub2()
{
   // self subtruction
   TStopwatch w;
   Vector v3;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 -= vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 -= v1     :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}


template<class Vector>
void TestVector<Vector>::Scale()
{
// normal multiply
   TStopwatch w;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 1; i< N; ++i) {
         Vector v3 = scale[i]*vlist[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 = A * v1 :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}

template<class Vector>
void TestVector<Vector>::Scale2()
{
   // self scale
   TStopwatch w;
   Vector v3;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 = vlist[i];
         v3 *= scale[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v *= a     :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}

template<class Vector>
void TestVector<Vector>::Divide()
{
// normal divide
   TStopwatch w;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 1; i< N; ++i) {
         Vector v3 = vlist[i]/scale[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v2 = v1 / a :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}

template<class Vector>
void TestVector<Vector>::Divide2()
{
   // self divide
   TStopwatch w;
   Vector v3;
   w.Start();
   double s = 0;
   for (int l = 0; l<100; ++l) {
      for (int i = 0; i< N; ++i) {
         v3 = vlist[i];
         v3 /= scale[i];
         s += VSUM(v3);
      }
   }

   std::cout << "Time for  v /= a      :\t" << w.RealTime() << "\t" << w.CpuTime() << std::endl;
   std::cout << "value " << s << std::endl << std::endl;
   fTime[fTest++] = w.CpuTime();
}


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
             << " v1 /= a   " << std::endl;

   for (int i = 0; i < fTest; ++i) {
      std::cout << std::setw(8) << fTime[i] << "   ";
   }
   std::cout << std::endl << std::endl;
}

int main() {
   TestVector<VecType> t;
#ifndef USE_POINT
   t.Add();
   t.Add2();
   t.Sub();
   t.Sub2();
#endif
   t.Scale();
   t.Scale2();
#ifndef USE_ROOT
   t.Divide();
   t.Divide2();
#endif

   // summurize test
   t.PrintSummary();
}
