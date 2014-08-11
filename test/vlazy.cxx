// @(#)root/test:$Id$
// Author: Fons Rademakers   14/11/97

//
// Sample code showing off a few advanced features
// and comparing them (time-wise) with traditional ones.
//
// Simple example: downsampling a matrix, that is, creating a matrix
// that is 4 times (twice in each dimension) smaller than the original
// matrix, by picking every other sample of the latter.
//

#include "TStopwatch.h"
#include "TMatrix.h"
#include "TMatrixFLazy.h"
#include "Riostream.h"


class do_downsample : public TElementPosActionF {
private:
   const TMatrix &fOrigMatrix;
   const int row_lwb, col_lwb;
   void Operation(Real_t &element) const
       { element = fOrigMatrix((fI-row_lwb)*2+row_lwb,(fJ-col_lwb)*2+col_lwb); }
public:
   do_downsample(const TMatrix &orig_matrix)
      : fOrigMatrix(orig_matrix),
        row_lwb(orig_matrix.GetRowLwb()),
        col_lwb(orig_matrix.GetColLwb()) { }
};

// Downsample matrix - new style
class downsample_matrix : public TMatrixFLazy {
private:
   const TMatrix &fOrigMatrix;
   void FillIn(TMatrixF &m) const;
public:
  downsample_matrix(const TMatrix &orig_matrix);
};

// Just figure out the dimensions of the downsampled (lazy) matrix
downsample_matrix::downsample_matrix(const TMatrix &orig_matrix)
        : TMatrixFLazy(orig_matrix.GetRowLwb(),
                      (orig_matrix.GetNrows()+1)/2 + orig_matrix.GetRowLwb()-1,
                      orig_matrix.GetColLwb(),
                      (orig_matrix.GetNcols()+1)/2 + orig_matrix.GetColLwb()-1),
fOrigMatrix(orig_matrix)
{ }

// "construct" the new matrix (when the lazy matrix is being "rolled out")
void downsample_matrix::FillIn(TMatrixF &m) const
{
   do_downsample d(fOrigMatrix);
   m.Apply(d);
}

// Downsample in the traditional style
static TMatrix traditional_downsampling(const TMatrix &orig_matrix)
{
   TMatrix smaller_m(orig_matrix.GetRowLwb(),
                     (orig_matrix.GetNrows()+1)/2 + orig_matrix.GetRowLwb()-1,
                     orig_matrix.GetColLwb(),
                     (orig_matrix.GetNcols()+1)/2 + orig_matrix.GetColLwb()-1);

   for (int i = 0; i < smaller_m.GetNrows(); i++)
      for (int j = 0; j < smaller_m.GetNcols(); j++)
         smaller_m(i+smaller_m.GetRowLwb(),j+smaller_m.GetColLwb()) =
         orig_matrix(2*i+smaller_m.GetRowLwb(),2*j+smaller_m.GetColLwb());
   return smaller_m;
}

int main()
{
   std::cout << "\nDownsample matrices using traditional and non-traditional methods"
        << std::endl;

   TStopwatch sw;

   {
      std::cout << "\nMake sure that both methods give the same results" << std::endl;
      TMatrix orig_m = THaarMatrixF(9,201);   // which is a pretty big matrix
      TMatrix small1 = traditional_downsampling(orig_m);
      TMatrix small2 = downsample_matrix(orig_m);
      R__ASSERT( small1 == small2 );
   }

   {
      std::cout << "\nClock the traditional downsampling" << std::endl;
      sw.Start();
      for (int order = 1; order <= 10; order++) {
         TMatrix orig_m = THaarMatrixF(order);   // may be pretty big, btw
         for (int count = 0; count < (1<<(12-order)); count++) {
            TMatrix small = traditional_downsampling(orig_m);
            small(0,0) = 1;                     // just to use the matrix
         }
      }
      std::cout << "\tIt took " << sw.RealTime()
           << " sec to complete the test" << std::endl;
   }

   {
      std::cout << "\nClock the 'new style' downsampling (with lazy matrices)"<< std::endl;
      sw.Start();
      for (int order = 1; order <= 10; order++) {
         TMatrix orig_m = THaarMatrixF(order);     // may be pretty big, btw
         for (int count = 0; count < (1<<(12-order)); count++) {
            TMatrix small = downsample_matrix(orig_m);
            small(0,0) = 1;                       // just to use the matrix
         }
      }
      std::cout << "\tIt took " << sw.RealTime()
           << " sec to complete the test" << std::endl;
   }
   return 0;
}
