#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <typeinfo>

#include <TMath.h>
#include <TError.h>

using std::cout, std::endl, std::vector, std::sort;
using namespace TMath;

bool showVector = true;

template <typename T>
void testNormCross()
{
   //Float_t NormCross(const Float_t v1[3],const Float_t v2[3],Float_t out[3])

   T fv1[] = {1,2,3};
   T fv2[] = {4,5,6};
   T fout[3];

   std::string type;
   if ( strcmp( typeid(T).name(), "f" ) == 0 )
      type = "Float_t";
   else if ( strcmp( typeid(T).name(), "d" ) == 0 )
      type = "Double_t";
   else
      type = typeid(T).name();

   TMath::NormCross(fv1,fv2,fout);

   cout << "NormCross(const " << type << " v1[3],const "
   << type << " v2[3]," << type << " out[3]): out = ["
   << fout[0] << ", " << fout[1] << ", " << fout[2] << "]"
   << endl;
}

template <typename T>
void testArrayDerivatives()
{
   const Long64_t n = 10;
   const double h = 0.1;
   T sa[n] = {18, 47, 183, 98, 56, 74, 28, 75, 10, 89};
   T *gradient = TMath::Gradient(n, sa, h);
   T *laplacian = TMath::Laplacian(n, sa, h);

   const T gradienta[n] = {290, 825, 255, -635, -120, -140, 5, -90, 70, 790};
   const T laplaciana[n] = {10875, 2675, -5525, 1075, 1500, -1600, 2325, -2800, 3600, 10000};

   // test results

   for (Long64_t i = 0; i < n; i++) {
      if (gradient[i] != gradienta[i])
         Error("testArrayDerivatives", "For Gradient, different values found at i = %lld", i);

      if (laplacian[i] != laplaciana[i])
         Error("testArrayDerivatives", "For Laplacian, different values found at i = %lld", i);
   }

   delete [] gradient;
   delete [] laplacian;

}

template <typename T, typename U>
void testArrayFunctions()
{
   const U n = 10;
   const U k = 3;
   U index[n];
   U is;

   T sa[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};

   if ( showVector )
   {
      cout << "Vector a[] = {" << sa[0];
      for ( Int_t i = 1; i < n; ++i )
         cout << ", " << sa[i];
      cout << "}\n" << endl;
      showVector = false;
   }

   cout << "Min: a[" << LocMin(n, sa) << "] = " << MinElement(n, sa)
        << " Max: a[" << LocMax(n, sa) << "] = " << MaxElement(n, sa)
        << " Mean: " << Mean(n, sa)
        << " GeomMean: " << GeomMean(n, sa)
        << " RMS: " << RMS(n, sa)
        << " Median: " << Median(n, sa)
        << " KOrdStat(3): " << KOrdStat(n, sa, k)
        << endl;

   Sort(n, sa, index, kFALSE);
   cout << "Sorted a[] = {" << sa[index[0]];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << sa[index[i]];
   cout << "}" << endl;

   sort(sa, sa+n);
   is = BinarySearch(n, sa, (T) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;
}

template <typename T>
void testIteratorFunctions()
{
   const Long64_t n = 10;
   vector<Int_t> index(n);
   Long64_t is;

   T tsa[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};
   vector<T> sa(n);
   for ( int i = 0;  i < n; ++i ) sa[i] = tsa[i];

   if ( showVector )
   {
      cout << "\nVector a[] = {" << sa[0];
      for ( Int_t i = 1; i < n; ++i )
         cout << ", " << sa[i];
      cout << "}\n" << endl;
      showVector = false;
   }

   cout << "Min: " << *LocMin(sa.begin(), sa.end())
        << " Max: " << *LocMax(sa.begin(), sa.end())
        << " Mean: " << Mean(sa.begin(), sa.end())
        << " GeomMean: " << GeomMean(sa.begin(), sa.end())
        << " RMS: " << RMS(sa.begin(), sa.end())
        << endl;

   TMath::SortItr(sa.begin(), sa.end(), index.begin(), kFALSE);
   cout << "Sorted a[] = {" << sa[ index[0] ];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << sa[ index[i] ];
   cout << "}" << endl;

   sort(&sa[0], &sa[0]+n);
   is = BinarySearch(n, &sa[0], (T) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;
}

template <typename T>
void testPoints(T x, T y)
{
   const Int_t n = 4;

   T dx[4] = {0, 0, 2, 2};
   T dy[4] = {0, 2, 2, 0};
   cout << "Point(" << x << "," << y << ") IsInside?: "
        << IsInside( x, y, n, dx, dy) << endl;

}

template <typename T>
void testPlane()
{
   T dp1[3] = {0,0,0};
   T dp2[3] = {1,0,0};
   T dp3[3] = {0,1,0};
   T dn[3];
   Normal2Plane(dp1, dp2, dp3, dn);
   cout << "Normal: ("
        << dn[0] << ", "
        << dn[1] << ", "
        << dn[2] << ")"
        << endl;
}

void testBreitWignerRelativistic()
{
  Double_t median = 5000;
  Double_t gamma = 100;
  Int_t nPoints = 10;
  Double_t xMinimum = 0; Double_t xMaximum = 10000;
  Double_t xStepSize = (xMaximum-xMinimum)/nPoints;

  for (Int_t i=0;i<=nPoints;i++) {
    Double_t currentX = xMinimum+i*xStepSize;
    cout << "BreitWignerRelativistic(" << currentX << "," << median << "," << gamma << ") = " << BreitWignerRelativistic(currentX,median,gamma) << endl;
  }
}

void testHalfSampleMode()
{
   // Let's compare the results with a completely independent implementation in MATLAB, see:
   // https://es.mathworks.com/matlabcentral/fileexchange/65579-ivim-model-fitting#functions_tab

   const long testdata_n = 50;
   double testdata[testdata_n] =
     {-1.8626292050574662, -1.2588261580948075, -1.2148747383283962, -0.88052174765194313, -0.85819166488158083,
      -0.70330273835955692,-0.62581680819121988, -0.60199237302865982, -0.056104207433014919, -0.048469532592846587,
      -0.045160899289979461, 0.00090904300338501276, 0.07801333189364168, 0.13270976192391337, 0.16400228933346139,
      0.20005259782423812, 0.63978982255456396, 0.67136185490980282, 0.68026440980784897, 0.7130887489094474,
      0.71656371678168229, 0.72363941257687625, 0.75056685823912761, 0.75558971232198124, 0.79884984432253403,
      0.8188843130575223, 0.83576827230628292, 0.84235396751558722, 0.85809047760873358, 0.89413700981004551,
      0.9040856696277757, 0.91910499099412069, 0.20717036575551762, 0.2161088542792593, 0.24097449144042737,
      0.24416261486911417, 0.36874321835120116, 0.37189144575412991, 0.3890935935298675, 0.40520928475289097,
      0.42184335741753576, 0.45308183033935723, 0.47769262360841214, 0.48905822724024939, 0.48979853918045224,
      0.53563861071255214, 0.61398403826022885, 0.62855905995409977, 0.92055153154640645, 0.9373728168229567};
   R__ASSERT(TMath::Abs(TMath::ModeHalfSample(testdata_n, testdata, nullptr) - 0.9198) < 1e-4);
   // Check equal weights is the same as no weights
   double testw[testdata_n]{}; // all equal zero weights
   R__ASSERT(TMath::ModeHalfSample(testdata_n, testdata, nullptr) ==
             TMath::ModeHalfSample(testdata_n, testdata, testw));

   const long testdata1_n = 5;
   unsigned short testdata1[testdata1_n] = {0, 2, 2, 1, 1};
   R__ASSERT(TMath::ModeHalfSample(testdata1_n, testdata1) == 1.);

   const long testdata2_n = 16;
   unsigned short testdata2[testdata2_n] = {0, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2};
   R__ASSERT(TMath::ModeHalfSample(testdata2_n, testdata2) == 0.);

   const long testdata3_n = 4;
   double testdata3[testdata3_n] = {1, 2, 3, 3.25};
   R__ASSERT(TMath::ModeHalfSample(testdata3_n, testdata3) == (3 + 3.25) / 2.0);
   // Check that the low-n cases work as expected.
   R__ASSERT(TMath::ModeHalfSample(1, testdata3) == 1.);
   R__ASSERT(TMath::ModeHalfSample(2, testdata3) == 1.5);
   R__ASSERT(TMath::ModeHalfSample(3, testdata3) == 2.);
   R__ASSERT(TMath::ModeHalfSample(3, testdata3 + 1) == (3 + 3.25) / 2.0);

   const long testdata4_n = 10;
   unsigned short testdata4[testdata4_n] = {1, 1, 1, 1, 0, 0, 0, 2, 2, 2};
   R__ASSERT(TMath::ModeHalfSample(testdata4_n, testdata4) == 0.);

   const long testdata5_n = 18;
   unsigned short testdata5[testdata5_n] = {1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2};
   R__ASSERT(TMath::ModeHalfSample(testdata5_n, testdata5) == 2.);

   // Test now with real weights
   const long testdata6_n = 4;
   short testdata6[testdata6_n] = {-1, 2, 3, 5};
   double weightdata6[testdata6_n] = {2, 3, 1, 6};
   R__ASSERT(TMath::ModeHalfSample(testdata6_n, testdata6, weightdata6) == 5);
   R__ASSERT(TMath::ModeHalfSample(2, testdata6, weightdata6) == TMath::Mean(2, testdata6, weightdata6));

   // Test now with real weights and duplicates
   const long testdata7_n = 5;
   short testdata7[testdata7_n] = {-1, 2, 3, 5, -1};
   double weightdata7[testdata7_n] = {2, 3, 1, 6, 5};
   R__ASSERT(TMath::ModeHalfSample(testdata7_n, testdata7, weightdata7) == -1);
}

void testTMath()
{
   cout << "Starting tests on TMath..." << endl;

   cout << "\nNormCross tests: " << endl;

   testNormCross<Float_t>();
   testNormCross<Double_t>();

   cout << "\nArray functions tests: " << endl;

   testArrayFunctions<Short_t,Long64_t>();
   testArrayFunctions<Int_t,Long64_t>();
   testArrayFunctions<Float_t,Long64_t>();
   testArrayFunctions<Double_t,Long64_t>();
   testArrayFunctions<Double_t,Int_t>();
   testArrayFunctions<Long_t,Long64_t>();
   testArrayFunctions<Long64_t,Long64_t>();

   cout << "\nArray derivative tests: " << endl;

   testArrayDerivatives<Short_t>();
   testArrayDerivatives<Int_t>();
   testArrayDerivatives<Float_t>();
   testArrayDerivatives<Double_t>();
   testArrayDerivatives<Long_t>();
   testArrayDerivatives<Long64_t>();

   cout << "\nIterator functions tests: " << endl;

   testIteratorFunctions<Short_t>();
   testIteratorFunctions<Int_t>();
   testIteratorFunctions<Float_t>();
   testIteratorFunctions<Double_t>();
   testIteratorFunctions<Long_t>();
   testIteratorFunctions<Long64_t>();

   cout << "\nPoint functions tests: " << endl;

   testPoints<Double_t>(1.3, 0.5);
   testPoints<Float_t>(-0.2, 1.7);
   testPoints<Int_t>(1, 1);

   cout << "\nPLane functions tests: " << endl;

   testPlane<Double_t>();
   testPlane<Float_t>();

   cout << "\nBreitWignerRelativistic tests: " << endl;

   testBreitWignerRelativistic();

   cout << "\nHalfSampleMode tests: " << endl;
   testHalfSampleMode();
}

int main()
{
   testTMath();

   return 0;
}
