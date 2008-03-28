#include <iostream>
#include <algorithm>

#include <TRandom2.h>
#include <TStopwatch.h>
#include <TMath.h>

using namespace std;
using namespace TMath;

const unsigned int SIZE = 100000;
const unsigned int NUMTEST = 500;

//#define DEBUG

template <typename T> T randD() {
   static TRandom2 r( time( 0 ) );
   return (T) r.Uniform(-500,500);
}

template <typename T> double stressVector(const char* type)
{
   cout << "Generating random vector of '" 
        << type << "'..." << endl;

   double totalTime = 0;

   T *vector = new T[SIZE];
   generate(vector, &vector[SIZE], randD<T>);

#ifdef DEBUG
   for ( unsigned int i = 0; i < SIZE; ++i )
      cout << vector[i] << " " << endl;
#endif

   TStopwatch w;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      MinElement(SIZE, vector);
   w.Stop();
   cout << "MinMaxElement() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      LocMin(SIZE, vector);
   w.Stop();
   cout << "LocMin/Max() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      Mean(SIZE, vector);
   w.Stop();
   cout << "Mean() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      Median(SIZE, vector);
   w.Stop();
   cout << "Median() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      RMS(SIZE, vector);
   w.Stop();
   cout << "RMS() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      GeomMean(SIZE, vector);
   w.Stop();
   cout << "GeomMean() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   Int_t index[SIZE];
   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      Sort(SIZE, vector, index, kFALSE);
   w.Stop();
   cout << "Sort() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   w.Start( kTRUE );
   for ( unsigned int i = 0; i < NUMTEST; ++i )
      BinarySearch(SIZE, vector, vector[1]);
   w.Stop();
   cout << "BinarySearch() Time: " << w.CpuTime()/NUMTEST << endl;
   totalTime += w.CpuTime()/NUMTEST;

   cout << "Total Time: " << totalTime << "\n" << endl;

   return totalTime;
}

void stressTMath() 
{
   double totalTime = 0;
   
   cout << "Stress Test Start..." << endl;

   totalTime += stressVector<Short_t>("Short_t");
   totalTime += stressVector<Int_t>("Int_t");
   totalTime += stressVector<Float_t>("Float_t");
   totalTime += stressVector<Double_t>("Double_t");
   totalTime += stressVector<Long_t>("Long_t");
   totalTime += stressVector<Long64_t>("Long64_t");
   
   cout << "Total Test Time: " << totalTime << "\n" << endl;

   cout << "End of Stress Test..." << endl;

   return;
}


int main()
{
   stressTMath();

   return 0;
}
