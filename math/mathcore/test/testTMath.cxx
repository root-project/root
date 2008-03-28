#include <iostream>

#include <TMath.h>

using namespace std;
using namespace TMath;

void testNormCross()
{
//Float_t NormCross(const Float_t v1[3],const Float_t v2[3],Float_t out[3])

   Float_t fv1[] = {1,2,3};
   Float_t fv2[] = {4,5,6};
   Float_t fout[3];

   TMath::NormCross(fv1,fv2,fout);
   cout << "NormCross(const Float_t v1[3],const Float_t v2[3],Float_t out[3]): out = [" 
        << fout[0] << ", " << fout[1] << ", " << fout[2] << "]"
        << endl;

   Double_t dv1[] = {1,2,3};
   Double_t dv2[] = {4,5,6};
   Double_t dout[3];

   TMath::NormCross(dv1,dv2,dout);
   cout << "NormCross(const Double_t v1[3],const Double_t v2[3],Double_t out[3]): out = [" 
        << dout[0] << ", " << dout[1] << ", " << dout[2] << "]\n"
        << endl;
}


void testArrayFunctions()
{
   const Long64_t n = 10;
   const Long64_t k = 3;
   Int_t index[n];
   Long64_t is;

// Short_t MinElement(Long64_t n, const Short_t *a)
   Short_t sa[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};

   cout << "Vector a[] = {" << sa[0];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << sa[i];
   cout << "}\n" << endl;

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
   is = BinarySearch(n, sa, (Short_t) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;

// Int_t MinElement(Long64_t n, const Int_t *a)
   Int_t ia[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};
   cout << "Min: a[" << LocMin(n, ia) << "] = " << MinElement(n, ia)
        << " Max: a[" << LocMax(n, ia) << "] = " << MaxElement(n, ia)
        << " Mean: " << Mean(n, ia)
        << " GeomMean: " << GeomMean(n, ia)
        << " RMS: " << RMS(n, ia)
        << " Median: " << Median(n, ia)
        << " KOrdStat(3): " << KOrdStat(n, sa, k)
        << endl;

   Sort(n, ia, index, kFALSE);
   cout << "Sorted a[] = {" << ia[index[0]];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << ia[index[i]];
   cout << "}" << endl;

   sort(sa, sa+n);
   is = BinarySearch(n, sa, (Short_t) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;

// Float_t MinElement(Long64_t n, const Float_t *a)
   Float_t fa[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};
   cout << "Min: a[" << LocMin(n, fa) << "] = " << MinElement(n, fa)
        << " Max: a[" << LocMax(n, fa) << "] = " << MaxElement(n, fa)
        << " Mean: " << Mean(n, fa)
        << " GeomMean: " << GeomMean(n, fa)
        << " RMS: " << RMS(n, fa)
        << " Median: " << Median(n, fa)
        << " KOrdStat(3): " << KOrdStat(n, sa, k)
        << endl;

   Sort(n, fa, index, kFALSE);
   cout << "Sorted a[] = {" << fa[index[0]];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << fa[index[i]];
   cout << "}" << endl;

   sort(sa, sa+n);
   is = BinarySearch(n, sa, (Short_t) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;

// Double_t MinElement(Long64_t n, const Double_t *a)
   Double_t da[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};
   cout << "Min: a[" << LocMin(n, da) << "] = " << MinElement(n, da)
        << " Max: a[" << LocMax(n, da) << "] = " << MaxElement(n, da)
        << " Mean: " << Mean(n, da)
        << " GeomMean: " << GeomMean(n, da)
        << " RMS: " << RMS(n, da)
        << " Median: " << Median(n, da)
        << " KOrdStat(3): " << KOrdStat(n, sa, k)
        << endl;

   Sort(n, da, index, kFALSE);
   cout << "Sorted a[] = {" << da[index[0]];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << da[index[i]];
   cout << "}" << endl;

   sort(sa, sa+n);
   is = BinarySearch(n, sa, (Short_t) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;

// Long_t MinElement(Long64_t n, const Long_t *a)
   Long_t la[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};
   cout << "Min: a[" << LocMin(n, la) << "] = " << MinElement(n, la)
        << " Max: a[" << LocMax(n, la) << "] = " << MaxElement(n, la)
        << " Mean: " << Mean(n, la)
        << " GeomMean: " << GeomMean(n, la)
        << " RMS: " << RMS(n, la)
        << " Median: " << Median(n, la)
        << " KOrdStat(3): " << KOrdStat(n, sa, k)
        << endl;

   Sort(n, la, index, kFALSE);
   cout << "Sorted a[] = {" << la[index[0]];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << la[index[i]];
   cout << "}" << endl;

   sort(sa, sa+n);
   is = BinarySearch(n, sa, (Short_t) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;

// Long64_t MinElement(Long64_t n, const Long64_t *a)
   Long64_t l64a[n] = { 2, 55 ,23, 57, -9, 24, 6, 82, -4, 10};
   cout << "Min: a[" << LocMin(n, l64a) << "] = " << MinElement(n, l64a)
        << " Max: a[" << LocMax(n, l64a) << "] = " << MaxElement(n, l64a)
        << " Mean: " << Mean(n, l64a)
        << " GeomMean: " << GeomMean(n, l64a)
        << " RMS: " << RMS(n, l64a)
        << " Median: " << Median(n, l64a)
        << " KOrdStat(3): " << KOrdStat(n, sa, k)
        << endl;

   Sort(n, l64a, index, kFALSE);
   cout << "Sorted a[] = {" << l64a[index[0]];
   for ( Int_t i = 1; i < n; ++i )
      cout << ", " << l64a[index[i]];
   cout << "}" << endl;

   sort(sa, sa+n);
   is = BinarySearch(n, sa, (Short_t) 57);
   cout << "BinarySearch(n, a, 57) = " << is << "\n" << endl;

}

void testPlane()
{
   const Int_t n = 4;

   Double_t dx[4] = {0.0, 0.0, 2.0, 2.0};
   Double_t dy[4] = {0.0, 2.0, 2.0, 0.0};
   cout << "Point(1.3,0.5) IsInside?: " << IsInside(1.3, 0.5, n, dx, dy) << endl;

   Float_t fx[4] = {0.0, 0.0, 2.0, 2.0};
   Float_t fy[4] = {0.0, 2.0, 2.0, 0.0};
   cout << "Point(-0.2f,1.7f) IsInside?: " << IsInside(-0.2f, 1.7f, n, fx, fy) << endl;

   Int_t ix[4] = {0, 0, 2, 2};
   Int_t iy[4] = {0, 2, 2, 0};
   cout << "Point(1,1) IsInside?: " << IsInside(1, 1, n, ix, iy) << endl;

   Double_t dp1[3] = {0,0,0};
   Double_t dp2[3] = {1,0,0};
   Double_t dp3[3] = {0,1,0};
   Double_t dn[3];
   Normal2Plane(dp1, dp2, dp3, dn);
   cout << "Normal: (" 
        << dn[0] << ", "
        << dn[1] << ", "
        << dn[2] << ")" 
        << endl;

   Double_t fp1[3] = {0,0,0};
   Double_t fp2[3] = {1,0,0};
   Double_t fp3[3] = {0,1,0};
   Double_t fn[3];
   Normal2Plane(fp1, fp2, fp3, fn);
   cout << "Normal: (" 
        << fn[0] << ", "
        << fn[1] << ", "
        << fn[2] << ")" 
        << endl;  

}


void testTMath() 
{
   cout << "Starting tests on TMath..." << endl;

   testNormCross();
   testArrayFunctions();
   testPlane();

}

int main()
{
   testTMath();

   return 0;
}
