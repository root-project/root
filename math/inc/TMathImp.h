
#ifndef __TMATHIMP__
#define __TMATHIMP__

template<typename T> 
struct CompareDesc { 

   CompareDesc(const T *  d) : fData(d) {}

   bool operator()(int i1, int i2) { 
      return fData[i1] > fData[i2];
   }

   const T * fData; 
};

template<typename T> 
struct CompareAsc { 

   CompareAsc(const T *  d) : fData(d) {}

   bool operator()(int i1, int i2) { 
      return fData[i1] < fData[i2];
   }

   const T * fData; 
};

template <typename T> Double_t TMath::Mean(Long64_t n, const T *a, const Double_t *w)
{
   // Return the weighted mean of an array a with length n.

   if (n <= 0 || !a) return 0;

   Double_t sum = 0;
   Double_t sumw = 0;
   if (w) {
      for (Long64_t i = 0; i < n; i++) {
         if (w[i] < 0) {
            ::Error("TMath::Mean","w[%d] = %.4e < 0 ?!",i,w[i]);
            return 0;
         }
         sum  += w[i]*a[i];
         sumw += w[i];
      }
      if (sumw <= 0) {
         ::Error("TMath::Mean","sum of weights == 0 ?!");
         return 0;
      }
   } else {
      sumw = n;
      for (Long64_t i = 0; i < n; i++)
         sum += a[i];
   }

   return sum/sumw;
}


template <typename T> Double_t TMath::GeomMean(Long64_t n, const T *a)
{
   // Return the geometric mean of an array a with length n.
   // geometric_mean = (Prod_i=0,n-1 |a[i]|)^1/n

   if (n <= 0 || !a) return 0;

   Double_t logsum = 0.;
   for (Long64_t i = 0; i < n; i++) {
      if (a[i] == 0) return 0.;
      Double_t absa = (Double_t) TMath::Abs(a[i]);
      logsum += TMath::Log(absa);
   }

   return TMath::Exp(logsum/n);
}

template <typename T> Double_t TMath::RMS(Long64_t n, const T *a)
{
   // Return the RMS of an array a with length n.

   if (n <= 0 || !a) return 0;

   Double_t tot = 0, tot2 =0, adouble;
   for (Long64_t i=0;i<n;i++) {
      adouble=Double_t(a[i]);
      tot += adouble; tot2 += adouble*adouble;
   }
   Double_t n1 = 1./n;
   Double_t mean = tot*n1;
   Double_t rms = TMath::Sqrt(TMath::Abs(tot2*n1 -mean*mean));
   return rms;
}

template <typename T> Double_t TMath::Median(Long64_t n, const T *a,  const Double_t *w, Long64_t *work)
{
   // Return the median of the array a where each entry i has weight w[i] .
   // Both arrays have a length of at least n . The median is a number obtained
   // from the sorted array a through
   //
   // median = (a[jl]+a[jh])/2.  where (using also the sorted index on the array w)
   //
   // sum_i=0,jl w[i] <= sumTot/2
   // sum_i=0,jh w[i] >= sumTot/2
   // sumTot = sum_i=0,n w[i]
   //
   // If w=0, the algorithm defaults to the median definition where it is
   // a number that divides the sorted sequence into 2 halves.
   // When n is odd or n > 1000, the median is kth element k = (n + 1) / 2.
   // when n is even and n < 1000the median is a mean of the elements k = n/2 and k = n/2 + 1.
   //
   // If work is supplied, it is used to store the sorting index and assumed to be
   // >= n . If work=0, local storage is used, either on the stack if n < kWorkMax
   // or on the heap for n >= kWorkMax .

   return MedianImp(n, a, w, work); 
}

template <typename T> Long64_t TMath::BinarySearch(Long64_t n, const T  *array, T value)
{
   // Binary search in an array of n values to locate value.
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.


#ifdef USE_NEW_STD_IMPL
   const T* pind;
   pind = std::lower_bound(array, array + n, value);
   Long64_t index = ((*pind == value)? (pind - array): ( pind - array - 1));

   return index;
#else

   Long64_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == array[middle-1]) return middle-1;
      if (value  < array[middle-1]) nabove = middle;
      else                          nbelow = middle;
   }
   return nbelow-1;

#endif
}

template <typename T> Long64_t TMath::BinarySearch(Long64_t n, const T **array, T value)
{
   // Binary search in an array of n values to locate value.
   //
   // Array is supposed  to be sorted prior to this call.
   // If match is found, function returns position of element.
   // If no match found, function gives nearest element smaller than value.

#ifdef USE_NEW_STD_IMPL
   const T* pind;
   pind = std::lower_bound(*array, *array + n, value);
   Long64_t index = ((*pind == value)? (pind - *array): ( pind - *array - 1));

   return index;
#else
   Long64_t nabove, nbelow, middle;
   nabove = n+1;
   nbelow = 0;
   while(nabove-nbelow > 1) {
      middle = (nabove+nbelow)/2;
      if (value == *array[middle-1]) return middle-1;
      if (value  < *array[middle-1]) nabove = middle;
      else                           nbelow = middle;
   }
   return nbelow-1;
#endif

}

template <typename Element, typename Index, typename Size> void TMath::Sort(Size n, const Element* a, Index* index, Bool_t down)
{
   // Sort the n1 elements of the Short_t array a.
   // In output the array index contains the indices of the sorted array.
   // If down is false sort in increasing order (default is decreasing order).
   // This is a translation of the CERNLIB routine sortzv (M101)
   // based on the quicksort algorithm.
   // NOTE that the array index must be created with a length >= n1
   // before calling this function.

   SortImp(n,a,index,down);
}

template <typename T> T *TMath::Cross(const T v1[3],const T v2[3], T out[3])
{
   // Calculate the Cross Product of two vectors:
   //         out = [v1 x v2]

   out[0] = v1[1] * v2[2] - v1[2] * v2[1];
   out[1] = v1[2] * v2[0] - v1[0] * v2[2];
   out[2] = v1[0] * v2[1] - v1[1] * v2[0];

   return out;
}

template <typename T> T * TMath::Normal2Plane(const T p1[3],const T p2[3],const T p3[3], T normal[3])
{
   // Calculate a normal vector of a plane.
   //
   //  Input:
   //     Float_t *p1,*p2,*p3  -  3 3D points belonged the plane to define it.
   //
   //  Return:
   //     Pointer to 3D normal vector (normalized)

   T v1[3], v2[3];

   v1[0] = p2[0] - p1[0];
   v1[1] = p2[1] - p1[1];
   v1[2] = p2[2] - p1[2];

   v2[0] = p3[0] - p1[0];
   v2[1] = p3[1] - p1[1];
   v2[2] = p3[2] - p1[2];

   NormCross(v1,v2,normal);
   return normal;
}

template <typename T> Bool_t TMath::IsInside(T xp, T yp, Int_t np, T *x, T *y) 
{
   // Function which returns kTRUE if point xp,yp lies inside the
   // polygon defined by the np points in arrays x and y, kFALSE otherwise
   // NOTE that the polygon must be a closed polygon (1st and last point
   // must be identical).

   Double_t xint;
   Int_t i;
   Int_t inter = 0;
   for (i=0;i<np-1;i++) {
      if (y[i] == y[i+1]) continue;
      if (yp <= y[i] && yp <= y[i+1]) continue;
      if (y[i] < yp && y[i+1] < yp) continue;
      xint = x[i] + (yp-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]);
      if ((Double_t)xp < xint) inter++;
   }
   if (inter%2) return kTRUE;
   return kFALSE;
}


template <class Element, class Index, class Size>
Double_t TMath::MedianImp(Size n, const Element *a,const Double_t *w, Index *work)
{
   // Return the median of the array a where each entry i has weight w[i] .
   // Both arrays have a length of at least n . The median is a number obtained
   // from the sorted array a through
   //
   // median = (a[jl]+a[jh])/2.  where (using also the sorted index on the array w)
   //
   // sum_i=0,jl w[i] <= sumTot/2
   // sum_i=0,jh w[i] >= sumTot/2
   // sumTot = sum_i=0,n w[i]
   //
   // If w=0, the algorithm defaults to the median definition where it is
   // a number that divides the sorted sequence into 2 halves.
   // When n is odd or n > 1000, the median is kth element k = (n + 1) / 2.
   // when n is even and n < 1000the median is a mean of the elements k = n/2 and k = n/2 + 1.
   //
   // If work is supplied, it is used to store the sorting index and assumed to be
   // >= n . If work=0, local storage is used, either on the stack if n < kWorkMax
   // or on the heap for n >= kWorkMax .

   const Int_t kWorkMax = 100;

   if (n <= 0 || !a) return 0;
   Bool_t isAllocated = kFALSE;
   Double_t median;
   Index *ind;
   Index workLocal[kWorkMax];

   if (work) {
      ind = work;
   } else {
      ind = workLocal;
      if (n > kWorkMax) {
         isAllocated = kTRUE;
         ind = new Index[n];
      }
   }

   if (w) {
      Double_t sumTot2 = 0;
      for (Int_t j = 0; j < n; j++) {
         if (w[j] < 0) {
            ::Error("TMath::Median","w[%d] = %.4e < 0 ?!",j,w[j]);
            return 0;
         }
         sumTot2 += w[j];
      }

      sumTot2 /= 2.;

      SortImp(n, a, ind, kFALSE);

      Double_t sum = 0.;
      Int_t jl;
      for (jl = 0; jl < n; jl++) {
         sum += w[ind[jl]];
         if (sum >= sumTot2) break;
      }

      Int_t jh;
      sum = 2.*sumTot2;
      for (jh = n-1; jh >= 0; jh--) {
         sum -= w[ind[jh]];
         if (sum <= sumTot2) break;
      }

      median = 0.5*(a[ind[jl]]+a[ind[jh]]);

   } else {

      if (n%2 == 1)
         median = KOrdStat(n, a,n/2, ind);
      else {
         median = 0.5*(KOrdStat(n, a, n/2 -1, ind)+KOrdStat(n, a, n/2, ind));
      }
   }

   if (isAllocated)
      delete [] ind;
   return median;
}

template <class Element, class Index, class Size>
void TMath::SortImp(Size n1, const Element *a,
                    Index *index, Bool_t down)
{
   // Templated version of the Sort.
   //
   // Sort the n1 elements of the array a.of Element
   // In output the array index contains the indices of the sorted array.
   // If down is false sort in increasing order (default is decreasing order).
   //
   // NOTE that the array index must be created with a length >= n1
   // before calling this function.
   //
   // See also the declarations at the top of this file.

    for(Size i = 0; i < n1; i++) { index[i] = i; }
    if ( down )
       std::sort(index, index + n1, CompareDesc<Element>(a) );
    else
       std::sort(index, index + n1, CompareAsc<Element>(a) );
}

template <class Element, typename Size>
Element TMath::KOrdStat(Size n, const Element *a, Size k, Size *work)
{
   // Returns k_th order statistic of the array a of size n
   // (k_th smallest element out of n elements).
   //
   // C-convention is used for array indexing, so if you want
   // the second smallest element, call KOrdStat(n, a, 1).
   //
   // If work is supplied, it is used to store the sorting index and
   // assumed to be >= n. If work=0, local storage is used, either on
   // the stack if n < kWorkMax or on the heap for n >= kWorkMax.
   //
   // Taken from "Numerical Recipes in C++" without the index array
   // implemented by Anna Khreshuk.
   //
   // See also the declarations at the top of this file

   const Int_t kWorkMax = 100;

   typedef Size Index;

   Bool_t isAllocated = kFALSE;
   Size i, ir, j, l, mid;
   Index arr;
   Index *ind;
   Index workLocal[kWorkMax];
   Index temp;

   if (work) {
      ind = work;
   } else {
      ind = workLocal;
      if (n > kWorkMax) {
         isAllocated = kTRUE;
         ind = new Index[n];
      }
   }

   for (Size ii=0; ii<n; ii++) {
      ind[ii]=ii;
   }
   Size rk = k;
   l=0;
   ir = n-1;
   for(;;) {
      if (ir<=l+1) { //active partition contains 1 or 2 elements
         if (ir == l+1 && a[ind[ir]]<a[ind[l]])
            {temp = ind[l]; ind[l]=ind[ir]; ind[ir]=temp;}
         Element tmp = a[ind[rk]];
         if (isAllocated)
            delete [] ind;
         return tmp;
      } else {
         mid = (l+ir) >> 1; //choose median of left, center and right
         {temp = ind[mid]; ind[mid]=ind[l+1]; ind[l+1]=temp;}//elements as partitioning element arr.
         if (a[ind[l]]>a[ind[ir]])  //also rearrange so that a[l]<=a[l+1]
            {temp = ind[l]; ind[l]=ind[ir]; ind[ir]=temp;}

         if (a[ind[l+1]]>a[ind[ir]])
            {temp=ind[l+1]; ind[l+1]=ind[ir]; ind[ir]=temp;}

         if (a[ind[l]]>a[ind[l+1]])
            {temp = ind[l]; ind[l]=ind[l+1]; ind[l+1]=temp;}

         i=l+1;        //initialize pointers for partitioning
         j=ir;
         arr = ind[l+1];
         for (;;){
            do i++; while (a[ind[i]]<a[arr]);
            do j--; while (a[ind[j]]>a[arr]);
            if (j<i) break;  //pointers crossed, partitioning complete
               {temp=ind[i]; ind[i]=ind[j]; ind[j]=temp;}
         }
         ind[l+1]=ind[j];
         ind[j]=arr;
         if (j>=rk) ir = j-1; //keep active the partition that
         if (j<=rk) l=i;      //contains the k_th element
      }
   }
}

#endif
