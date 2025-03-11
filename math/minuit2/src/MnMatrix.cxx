// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include <Minuit2/MnMatrix.h>

#include <cmath>
#include <numeric>

constexpr int PRECISION = 10;
constexpr int WIDTH = PRECISION + 7;

namespace {

/** Inverts a symmetric matrix. Matrix is first scaled to have all ones on
    the diagonal (equivalent to change of units) but no pivoting is done
    since matrix is positive-definite.
 */

int mnvert(ROOT::Minuit2::LASymMatrix &a)
{
   unsigned int nrow = a.Nrow();
   ROOT::Minuit2::LAVector s(nrow);
   ROOT::Minuit2::LAVector q(nrow);
   ROOT::Minuit2::LAVector pp(nrow);

   for (unsigned int i = 0; i < nrow; i++) {
      double si = a(i, i);
      if (si < 0.)
         return 1;
      s(i) = 1. / std::sqrt(si);
   }

   for (unsigned int i = 0; i < nrow; i++)
      for (unsigned int j = i; j < nrow; j++)
         a(i, j) *= (s(i) * s(j));

   for (unsigned i = 0; i < nrow; i++) {
      unsigned int k = i;
      if (a(k, k) == 0.)
         return 1;
      q(k) = 1. / a(k, k);
      pp(k) = 1.;
      a(k, k) = 0.;
      unsigned int kp1 = k + 1;
      if (k != 0) {
         for (unsigned int j = 0; j < k; j++) {
            pp(j) = a(j, k);
            q(j) = a(j, k) * q(k);
            a(j, k) = 0.;
         }
      }
      if (k != nrow - 1) {
         for (unsigned int j = kp1; j < nrow; j++) {
            pp(j) = a(k, j);
            q(j) = -a(k, j) * q(k);
            a(k, j) = 0.;
         }
      }
      for (unsigned int j = 0; j < nrow; j++)
         for (k = j; k < nrow; k++)
            a(j, k) += (pp(j) * q(k));
   }

   for (unsigned int j = 0; j < nrow; j++)
      for (unsigned int k = j; k < nrow; k++)
         a(j, k) *= (s(j) * s(k));

   return 0;
}

} // namespace

namespace ROOT {

namespace Minuit2 {

inline double sum_of_elements(double const *arr, unsigned int n)
{
   double out = 0.0;
   for (unsigned int i = 0; i < n; ++i) {
      out += std::abs(arr[i]);
   }
   return out;
}

double sum_of_elements(const LAVector &v)
{
   // calculate the absolute sum of the vector elements
   return sum_of_elements(v.Data(), v.size());
}

double sum_of_elements(const LASymMatrix &m)
{
   // calculate the absolute sum of all the matrix elements
   return sum_of_elements(m.Data(), m.size());
}

// Updates A := alpha*x*x' + A, where ap stores the upper triangle of A
void mndspr(unsigned int n, double alpha, const double *x, double *ap)
{
   /* System generated locals */
   int i__1, i__2;

   /* Local variables */
   double temp;
   int i__, j, k;
   int kk;

   /* Parameter adjustments */
   --ap;
   --x;

   /*     Quick return if possible. */

   if (n == 0 || alpha == 0.) {
      return;
   }

   /*     Start the operations. In this version the Elements of the array AP */
   /*     are accessed sequentially with one pass through AP. */

   kk = 1;

   /*        Form  A  when Upper triangle is stored in AP. */

   i__1 = n;
   for (j = 1; j <= i__1; ++j) {
      if (x[j] != 0.) {
         temp = alpha * x[j];
         k = kk;
         i__2 = j;
         for (i__ = 1; i__ <= i__2; ++i__) {
            ap[k] += x[i__] * temp;
            ++k;
         }
      }
      kk += j;
   }
}

// Updates y := alpha*A*x + beta*y, where ap stores the upper triangle of A
void Mndspmv(unsigned int n, double alpha, const double *ap, const double *x, double beta, double *y)
{
   /* System generated locals */
   int i__1, i__2;

   /* Local variables */
   double temp1, temp2;
   int i__, j, k;
   int kk;

   /* Parameter adjustments */
   --y;
   --x;
   --ap;

   /*     Quick return if possible. */

   if ((n == 0) || (alpha == 0. && beta == 1.)) {
      return;
   }

   /*     Set up the start points in  X  and  Y. */

   /*     Start the operations. In this version the Elements of the array AP */
   /*     are accessed sequentially with one pass through AP. */

   /*     First form  y := beta*y. */

   if (beta != 1.) {
      if (beta == 0.) {
         i__1 = n;
         for (i__ = 1; i__ <= i__1; ++i__) {
            y[i__] = 0.;
            /* L10: */
         }
      } else {
         i__1 = n;
         for (i__ = 1; i__ <= i__1; ++i__) {
            y[i__] = beta * y[i__];
            /* L20: */
         }
      }
   }
   if (alpha == 0.) {
      return;
   }
   kk = 1;

   /*        Form  y  when AP contains the Upper triangle. */

   i__1 = n;
   for (j = 1; j <= i__1; ++j) {
      temp1 = alpha * x[j];
      temp2 = 0.;
      k = kk;
      i__2 = j - 1;
      for (i__ = 1; i__ <= i__2; ++i__) {
         y[i__] += temp1 * ap[k];
         temp2 += ap[k] * x[i__];
         ++k;
      }
      y[j] = y[j] + temp1 * ap[kk + j - 1] + alpha * temp2;
      kk += j;
   }
}

LASymMatrix::LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double> &out)
   : fSize(0), fNRow(0), fData(nullptr)
{
   // constructor from expression based on outer product of symmetric matrices
   //   std::cout<<"LASymMatrix::LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>,
   //   double>& out)"<<std::endl;
   fNRow = out.Obj().Obj().Obj().size();
   fSize = fNRow * (fNRow + 1) / 2;
   fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
   memset(fData, 0, fSize * sizeof(double));
   Outer_prod(*this, out.Obj().Obj().Obj(), out.f() * out.Obj().Obj().f() * out.Obj().Obj().f());
}

LASymMatrix &
LASymMatrix::operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double> &out)
{
   // assignment operator from expression based on outer product of symmetric matrices
   //   std::cout<<"LASymMatrix& LASymMatrix::operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector,
   //   double>, double>, double>& out)"<<std::endl;
   if (fSize == 0 && fData == nullptr) {
      fNRow = out.Obj().Obj().Obj().size();
      fSize = fNRow * (fNRow + 1) / 2;
      fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
      memset(fData, 0, fSize * sizeof(double));
      Outer_prod(*this, out.Obj().Obj().Obj(), out.f() * out.Obj().Obj().f() * out.Obj().Obj().f());
   } else {
      LASymMatrix tmp(out.Obj().Obj().Obj().size());
      Outer_prod(tmp, out.Obj().Obj().Obj());
      tmp *= double(out.f() * out.Obj().Obj().f() * out.Obj().Obj().f());
      assert(fSize == tmp.size());
      memcpy(fData, tmp.Data(), fSize * sizeof(double));
   }
   return *this;
}

void Outer_prod(LASymMatrix &A, const LAVector &v, double f)
{
   // function performing outer product and adding it to A
   mndspr(v.size(), f, v.Data(), A.Data());
}

void Mndaxpy(unsigned int n, double da, const double *dx, double *dy)
{
   /* System generated locals */
   int i__1;

   /* Local variables */
   int i__, m, mp1;

   /* Parameter adjustments */
   --dy;
   --dx;

   /* Function Body */
   if (n <= 0) {
      return;
   }
   if (da == 0.) {
      return;
   }
   m = n % 4;
   if (m != 0) {
      i__1 = m;
      for (i__ = 1; i__ <= i__1; ++i__) {
         dy[i__] += da * dx[i__];
      }
      if (n < 4) {
         return;
      }
   }
   mp1 = m + 1;
   i__1 = n;
   for (i__ = mp1; i__ <= i__1; i__ += 4) {
      dy[i__] += da * dx[i__];
      dy[i__ + 1] += da * dx[i__ + 1];
      dy[i__ + 2] += da * dx[i__ + 2];
      dy[i__ + 3] += da * dx[i__ + 3];
   }
}

// symmetric matrix (positive definite only)

int Invert(LASymMatrix &t)
{
   // function for inversion of symmetric matrices using  mnvert function
   // (from Fortran Minuit)

   int ifail = 0;

   if (t.size() == 1) {
      double tmp = t.Data()[0];
      if (!(tmp > 0.))
         ifail = 1;
      else
         t.Data()[0] = 1. / tmp;
   } else {
      ifail = mnvert(t);
   }

   return ifail;
}

double inner_product(const LAVector &v1, const LAVector &v2)
{
   // calculate inner (dot) product of two vectors
   return std::inner_product(v1.Data(), v1.Data() + v1.size(), v2.Data(), 0.0);
}

int mneigen(double *a, unsigned int ndima, unsigned int n, unsigned int mits, double *work)
{
   constexpr double precis = 1.e-6;

   // compute matrix eignevalues (translation from mneig.F of Minuit)

   /* System generated locals */
   unsigned int a_dim1, a_offset, i__1, i__2, i__3;
   double r__1, r__2;

   /* Local variables */
   double b, c__, f, h__;
   unsigned int i__, j, k, l, m = 0;
   double r__, s;
   unsigned int i0, i1, j1, m1, n1;
   double hh, gl, pr, pt;

   /*          PRECIS is the machine precision EPSMAC */
   /* Parameter adjustments */
   a_dim1 = ndima;
   a_offset = 1 + a_dim1 * 1;
   a -= a_offset;
   --work;

   /* Function Body */
   int ifault = 1;

   i__ = n;
   i__1 = n;
   for (i1 = 2; i1 <= i__1; ++i1) {
      l = i__ - 2;
      f = a[i__ + (i__ - 1) * a_dim1];
      gl = (double)0.;

      if (l < 1) {
         goto L25;
      }

      i__2 = l;
      for (k = 1; k <= i__2; ++k) {
         /* Computing 2nd power */
         r__1 = a[i__ + k * a_dim1];
         gl += r__1 * r__1;
      }
   L25:
      /* Computing 2nd power */
      r__1 = f;
      h__ = gl + r__1 * r__1;

      if (gl > (double)1e-35) {
         goto L30;
      }

      work[i__] = (double)0.;
      work[n + i__] = f;
      goto L65;
   L30:
      ++l;

      gl = std::sqrt(h__);

      if (f >= (double)0.) {
         gl = -gl;
      }

      work[n + i__] = gl;
      h__ -= f * gl;
      a[i__ + (i__ - 1) * a_dim1] = f - gl;
      f = (double)0.;
      i__2 = l;
      for (j = 1; j <= i__2; ++j) {
         a[j + i__ * a_dim1] = a[i__ + j * a_dim1] / h__;
         gl = (double)0.;
         i__3 = j;
         for (k = 1; k <= i__3; ++k) {
            gl += a[j + k * a_dim1] * a[i__ + k * a_dim1];
         }

         if (j >= l) {
            goto L47;
         }

         j1 = j + 1;
         i__3 = l;
         for (k = j1; k <= i__3; ++k) {
            gl += a[k + j * a_dim1] * a[i__ + k * a_dim1];
         }
      L47:
         work[n + j] = gl / h__;
         f += gl * a[j + i__ * a_dim1];
      }
      hh = f / (h__ + h__);
      i__2 = l;
      for (j = 1; j <= i__2; ++j) {
         f = a[i__ + j * a_dim1];
         gl = work[n + j] - hh * f;
         work[n + j] = gl;
         i__3 = j;
         for (k = 1; k <= i__3; ++k) {
            a[j + k * a_dim1] = a[j + k * a_dim1] - f * work[n + k] - gl * a[i__ + k * a_dim1];
         }
      }
      work[i__] = h__;
   L65:
      --i__;
   }
   work[1] = (double)0.;
   work[n + 1] = (double)0.;
   i__1 = n;
   for (i__ = 1; i__ <= i__1; ++i__) {
      l = i__ - 1;

      if (work[i__] == (double)0. || l == 0) {
         goto L100;
      }

      i__3 = l;
      for (j = 1; j <= i__3; ++j) {
         gl = (double)0.;
         i__2 = l;
         for (k = 1; k <= i__2; ++k) {
            gl += a[i__ + k * a_dim1] * a[k + j * a_dim1];
         }
         i__2 = l;
         for (k = 1; k <= i__2; ++k) {
            a[k + j * a_dim1] -= gl * a[k + i__ * a_dim1];
         }
      }
   L100:
      work[i__] = a[i__ + i__ * a_dim1];
      a[i__ + i__ * a_dim1] = (double)1.;

      if (l == 0) {
         goto L110;
      }

      i__2 = l;
      for (j = 1; j <= i__2; ++j) {
         a[i__ + j * a_dim1] = (double)0.;
         a[j + i__ * a_dim1] = (double)0.;
      }
   L110:;
   }

   n1 = n - 1;
   i__1 = n;
   for (i__ = 2; i__ <= i__1; ++i__) {
      i0 = n + i__ - 1;
      work[i0] = work[i0 + 1];
   }
   work[n + n] = (double)0.;
   b = (double)0.;
   f = (double)0.;
   i__1 = n;
   for (l = 1; l <= i__1; ++l) {
      j = 0;
      h__ = precis * ((r__1 = work[l], std::fabs(r__1)) + (r__2 = work[n + l], std::fabs(r__2)));

      if (b < h__) {
         b = h__;
      }

      i__2 = n;
      for (m1 = l; m1 <= i__2; ++m1) {
         m = m1;

         if ((r__1 = work[n + m], std::fabs(r__1)) <= b) {
            goto L150;
         }
      }

   L150:
      if (m == l) {
         goto L205;
      }

   L160:
      if (j == mits) {
         return ifault;
      }

      ++j;
      pt = (work[l + 1] - work[l]) / (work[n + l] * (double)2.);
      r__ = std::sqrt(pt * pt + (double)1.);
      pr = pt + r__;

      if (pt < (double)0.) {
         pr = pt - r__;
      }

      h__ = work[l] - work[n + l] / pr;
      i__2 = n;
      for (i__ = l; i__ <= i__2; ++i__) {
         work[i__] -= h__;
      }
      f += h__;
      pt = work[m];
      c__ = (double)1.;
      s = (double)0.;
      m1 = m - 1;
      i__ = m;
      i__2 = m1;
      for (i1 = l; i1 <= i__2; ++i1) {
         j = i__;
         --i__;
         gl = c__ * work[n + i__];
         h__ = c__ * pt;

         if (std::fabs(pt) >= (r__1 = work[n + i__], std::fabs(r__1))) {
            goto L180;
         }

         c__ = pt / work[n + i__];
         r__ = std::sqrt(c__ * c__ + (double)1.);
         work[n + j] = s * work[n + i__] * r__;
         s = (double)1. / r__;
         c__ /= r__;
         goto L190;
      L180:
         c__ = work[n + i__] / pt;
         r__ = std::sqrt(c__ * c__ + (double)1.);
         work[n + j] = s * pt * r__;
         s = c__ / r__;
         c__ = (double)1. / r__;
      L190:
         pt = c__ * work[i__] - s * gl;
         work[j] = h__ + s * (c__ * gl + s * work[i__]);
         i__3 = n;
         for (k = 1; k <= i__3; ++k) {
            h__ = a[k + j * a_dim1];
            a[k + j * a_dim1] = s * a[k + i__ * a_dim1] + c__ * h__;
            a[k + i__ * a_dim1] = c__ * a[k + i__ * a_dim1] - s * h__;
         }
      }
      work[n + l] = s * pt;
      work[l] = c__ * pt;

      if ((r__1 = work[n + l], std::fabs(r__1)) > b) {
         goto L160;
      }

   L205:
      work[l] += f;
   }
   i__1 = n1;
   for (i__ = 1; i__ <= i__1; ++i__) {
      k = i__;
      pt = work[i__];
      i1 = i__ + 1;
      i__3 = n;
      for (j = i1; j <= i__3; ++j) {

         if (work[j] >= pt) {
            goto L220;
         }

         k = j;
         pt = work[j];
      L220:;
      }

      if (k == i__) {
         goto L240;
      }

      work[k] = work[i__];
      work[i__] = pt;
      i__3 = n;
      for (j = 1; j <= i__3; ++j) {
         pt = a[j + i__ * a_dim1];
         a[j + i__ * a_dim1] = a[j + k * a_dim1];
         a[j + k * a_dim1] = pt;
      }
   L240:;
   }
   ifault = 0;

   return ifault;
} /* mneig_ */

LAVector eigenvalues(const LASymMatrix &mat)
{
   // calculate eigenvalues of symmetric matrices using mneigen function (translate from fortran Minuit)
   unsigned int nrow = mat.Nrow();

   LAVector tmp(nrow * nrow);
   LAVector work(2 * nrow);

   for (unsigned int i = 0; i < nrow; i++) {
      for (unsigned int j = 0; j <= i; j++) {
         tmp(i + j * nrow) = mat(i, j);
         tmp(i * nrow + j) = mat(i, j);
      }
   }

   int info = mneigen(tmp.Data(), nrow, nrow, work.size(), work.Data());
   (void)info;
   assert(info == 0);

   LAVector result(nrow);
   for (unsigned int i = 0; i < nrow; i++) {
      result(i) = work(i);
   }

   return result;
}

double similarity(const LAVector &avec, const LASymMatrix &mat)
{
   // calculate the similarity vector-matrix product: V^T M V
   // use matrix product and then dot function

   LAVector tmp = mat * avec;

   return std::inner_product(avec.Data(), avec.Data() + avec.size(), tmp.Data(), 0.0);
}

void Mndscal(unsigned int n, double da, double *dx)
{
   for (unsigned int i = 0; i < n; ++i) {
      dx[i] *= da;
   }
}

thread_local int gMaxNP = 10;

int MnMatrix::SetMaxNP(int value)
{
   std::swap(gMaxNP, value);
   return value;
}

int MnMatrix::MaxNP()
{
   return gMaxNP;
}

std::ostream &operator<<(std::ostream &os, const LASymMatrix &matrix)
{
   // print a matrix
   const int pr = os.precision(8);
   const unsigned int nrow = matrix.Nrow();
   const unsigned int n = std::min(nrow, static_cast<unsigned int>(MnMatrix::MaxNP()));
   for (unsigned int i = 0; i < nrow; i++) {
      os << "\n";
      if (i == 0)
         os << "[[";
      else {
         if (i >= n) {
            os << "....\n";
            i = nrow - 1;
         }
         os << " [";
      }
      for (unsigned int j = 0; j < nrow; j++) {
         if (j >= n) {
            os << ".... ";
            j = nrow - 1;
         }
         os.width(15);
         os << matrix(i, j);
      }
      os << "]";
   }
   os << "]]";
   os.precision(pr);
   return os;
}

std::ostream &operator<<(std::ostream &os, const LAVector &vec)
{
   // print a vector
   const int pr = os.precision(PRECISION);
   const unsigned int nrow = vec.size();
   const unsigned int np = std::min(nrow, static_cast<unsigned int>(MnMatrix::MaxNP()));
   os << "\t[";
   for (unsigned int i = 0; i < np; i++) {
      os.width(WIDTH);
      os << vec(i);
   }
   if (np < nrow) {
      os << ".... ";
      os.width(WIDTH);
      os << vec(nrow - 1);
   }
   os << "]\t";
   os.precision(pr);
   return os;
}

} // namespace Minuit2

} // namespace ROOT
