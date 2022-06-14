// @(#)root/smatrix:$Id$
// Author: M. Schiller    2009

#ifndef ROOT_Math_CholeskyDecomp
#define ROOT_Math_CholeskyDecomp

/** @file
 * header file containing the templated implementation of matrix inversion
 * routines for use with ROOT's SMatrix classes (symmetric positive
 * definite case)
 *
 * @author Manuel Schiller
 * @date Aug 29 2008
 *    initial release inside LHCb
 * @date May 7 2009
 * factored code to provide a nice Cholesky decomposition class, along
 * with separate methods for solving a single linear system and to
 * obtain the inverse matrix from the decomposition
 * @date July 15th 2013
 *    provide a version of that class which works if the dimension of the
 *    problem is only known at run time
 * @date September 30th 2013
 *    provide routines to access the result of the decomposition L and its
 *    inverse
 */

#include <cmath>
#include <algorithm>

namespace ROOT {

   namespace Math {

/// helpers for CholeskyDecomp
namespace CholeskyDecompHelpers {
   // forward decls
   template<class F, class M> struct _decomposerGenDim;
   template<class F, unsigned N, class M> struct _decomposer;
   template<class F, class M> struct _inverterGenDim;
   template<class F, unsigned N, class M> struct _inverter;
   template<class F, class V> struct _solverGenDim;
   template<class F, unsigned N, class V> struct _solver;
   template<typename G> class PackedArrayAdapter;
}

/// class to compute the Cholesky decomposition of a matrix
/** class to compute the Cholesky decomposition of a symmetric
 * positive definite matrix
 *
 * provides routines to check if the decomposition succeeded (i.e. if
 * matrix is positive definite and non-singular), to solve a linear
 * system for the given matrix and to obtain its inverse
 *
 * the actual functionality is implemented in templated helper
 * classes which have specializations for dimensions N = 1 to 6
 * to achieve a gain in speed for common matrix sizes
 *
 * usage example:
 * @code
 * // let m be a symmetric positive definite SMatrix (use type float
 * // for internal computations, matrix size is 4x4)
 * CholeskyDecomp<float, 4> decomp(m);
 * // check if the decomposition succeeded
 * if (!decomp) {
 *   std::cerr << "decomposition failed!" << std::endl;
 * } else {
 *   // let rhs be a vector; we seek a vector x such that m * x = rhs
 *   decomp.Solve(rhs);
 *   // rhs now contains the solution we are looking for
 *
 *   // obtain the inverse of m, put it into m itself
 *   decomp.Invert(m);
 * }
 * @endcode
 */
template<class F, unsigned N> class CholeskyDecomp
{
private:
   /// lower triangular matrix L
   /** lower triangular matrix L, packed storage, with diagonal
    * elements pre-inverted */
   F fL[N * (N + 1) / 2];
   /// flag indicating a successful decomposition
   bool fOk;
public:
   /// perform a Cholesky decomposition
   /** perfrom a Cholesky decomposition of a symmetric positive
    * definite matrix m
    *
    * this is the constructor to uses with an SMatrix (and objects
    * that behave like an SMatrix in terms of using
    * operator()(int i, int j) for access to elements)
    */
   template<class M> CholeskyDecomp(const M& m) :
      fL( ), fOk(false)
   {
      using CholeskyDecompHelpers::_decomposer;
      fOk = _decomposer<F, N, M>()(fL, m);
   }

   /// perform a Cholesky decomposition
   /** perfrom a Cholesky decomposition of a symmetric positive
    * definite matrix m
    *
    * this is the constructor to use in special applications where
    * plain arrays are used
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(i,j) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> CholeskyDecomp(G* m) :
      fL(), fOk(false)
   {
      using CholeskyDecompHelpers::_decomposer;
      using CholeskyDecompHelpers::PackedArrayAdapter;
      fOk = _decomposer<F, N, PackedArrayAdapter<G> >()(
         fL, PackedArrayAdapter<G>(m));
   }

   /// returns true if decomposition was successful
   /** @returns true if decomposition was successful */
   bool ok() const { return fOk; }
   /// returns true if decomposition was successful
   /** @returns true if decomposition was successful */
   operator bool() const { return fOk; }

   /** @brief solves a linear system for the given right hand side
    *
    * Note that you can use both SVector classes and plain arrays for
    * rhs. (Make sure that the sizes match!). It will work with any vector
    * implementing the operator [i]
    *
    * @returns if the decomposition was successful
    */
   template<class V> bool Solve(V& rhs) const
   {
      using CholeskyDecompHelpers::_solver;
      if (fOk) _solver<F,N,V>()(rhs, fL);
      return fOk;
   }

   /** @brief place the inverse into m
    *
    * This is the method to use with an SMatrix.
    *
    * @returns if the decomposition was successful
    */
   template<class M> bool Invert(M& m) const
   {
      using CholeskyDecompHelpers::_inverter;
      if (fOk) _inverter<F,N,M>()(m, fL);
      return fOk;
   }

   /** @brief place the inverse into m
    *
    * This is the method to use with a plain array.
    *
    * @returns if the decomposition was successful
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(i,j) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> bool Invert(G* m) const
   {
      using CholeskyDecompHelpers::_inverter;
      using CholeskyDecompHelpers::PackedArrayAdapter;
      if (fOk) {
         PackedArrayAdapter<G> adapted(m);
         _inverter<F,N,PackedArrayAdapter<G> >()(adapted, fL);
      }
      return fOk;
   }

   /** @brief obtain the decomposed matrix L
    *
    * This is the method to use with a plain array.
    *
    * @returns if the decomposition was successful
    */
   template<class M> bool getL(M& m) const
   {
      if (!fOk) return false;
      for (unsigned i = 0; i < N; ++i) {
         // zero upper half of matrix
         for (unsigned j = i + 1; j < N; ++j)
         m(i, j) = F(0);
         // copy the rest
         for (unsigned j = 0; j <= i; ++j)
         m(i, j) = fL[i * (i + 1) / 2 + j];
         // adjust the diagonal - we save 1/L(i, i) in that position, so
         // convert to what caller expects
         m(i, i) = F(1) / m(i, i);
      }
      return true;
   }

   /** @brief obtain the decomposed matrix L
    *
    * @returns if the decomposition was successful
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(i,j) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> bool getL(G* m) const
   {
      if (!fOk) return false;
      // copy L
      for (unsigned i = 0; i < (N * (N + 1)) / 2; ++i)
         m[i] = fL[i];
      // adjust diagonal - we save 1/L(i, i) in that position, so convert to
      // what caller expects
      for (unsigned i = 0; i < N; ++i)
         m[(i * (i + 1)) / 2 + i] = F(1) / fL[(i * (i + 1)) / 2 + i];
      return true;
   }

   /** @brief obtain the inverse of the decomposed matrix L
    *
    * This is the method to use with a plain array.
    *
    * @returns if the decomposition was successful
    */
   template<class M> bool getLi(M& m) const
   {
      if (!fOk) return false;
      for (unsigned i = 0; i < N; ++i) {
         // zero lower half of matrix
         for (unsigned j = i + 1; j < N; ++j)
            m(j, i) = F(0);
         // copy the rest
         for (unsigned j = 0; j <= i; ++j)
         m(j, i) = fL[i * (i + 1) / 2 + j];
      }
      // invert the off-diagonal part of what we just copied
      for (unsigned i = 1; i < N; ++i) {
         for (unsigned j = 0; j < i; ++j) {
            typename M::value_type tmp = F(0);
            for (unsigned k = i; k-- > j;)
               tmp -= m(k, i) * m(j, k);
            m(j, i) = tmp * m(i, i);
         }
      }
      return true;
   }

   /** @brief obtain the inverse of the decomposed matrix L
    *
    * @returns if the decomposition was successful
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(j,i) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> bool getLi(G* m) const
   {
      if (!fOk) return false;
      // copy L
      for (unsigned i = 0; i < (N * (N + 1)) / 2; ++i)
         m[i] = fL[i];
      // invert the off-diagonal part of what we just copied
      G* base1 = &m[1];
      for (unsigned i = 1; i < N; base1 += ++i) {
         for (unsigned j = 0; j < i; ++j) {
            G tmp = F(0);
            const G *base2 = &m[(i * (i - 1)) / 2];
            for (unsigned k = i; k-- > j; base2 -= k)
            tmp -= base1[k] * base2[j];
            base1[j] = tmp * base1[i];
         }
      }
      return true;
   }
};

/// class to compute the Cholesky decomposition of a matrix
/** class to compute the Cholesky decomposition of a symmetric
 * positive definite matrix when the dimensionality of the problem is not known
 * at compile time
 *
 * provides routines to check if the decomposition succeeded (i.e. if
 * matrix is positive definite and non-singular), to solve a linear
 * system for the given matrix and to obtain its inverse
 *
 * the actual functionality is implemented in templated helper
 * classes which have specializations for dimensions N = 1 to 6
 * to achieve a gain in speed for common matrix sizes
 *
 * usage example:
 * @code
 * // let m be a symmetric positive definite SMatrix (use type float
 * // for internal computations, matrix size is 4x4)
 * CholeskyDecompGenDim<float> decomp(4, m);
 * // check if the decomposition succeeded
 * if (!decomp) {
 *   std::cerr << "decomposition failed!" << std::endl;
 * } else {
 *   // let rhs be a vector; we seek a vector x such that m * x = rhs
 *   decomp.Solve(rhs);
 *   // rhs now contains the solution we are looking for
 *
 *   // obtain the inverse of m, put it into m itself
 *   decomp.Invert(m);
 * }
 * @endcode
 */
template<class F> class CholeskyDecompGenDim
{
private:
   /** @brief dimensionality
    * dimensionality of the problem */
   unsigned fN;
   /// lower triangular matrix L
   /** lower triangular matrix L, packed storage, with diagonal
    * elements pre-inverted */
   F *fL;
   /// flag indicating a successful decomposition
   bool fOk;
public:
   /// perform a Cholesky decomposition
   /** perfrom a Cholesky decomposition of a symmetric positive
    * definite matrix m
    *
    * this is the constructor to uses with an SMatrix (and objects
    * that behave like an SMatrix in terms of using
    * operator()(int i, int j) for access to elements)
    */
   template<class M> CholeskyDecompGenDim(unsigned N, const M& m) :
      fN(N), fL(new F[(fN * (fN + 1)) / 2]), fOk(false)
   {
      using CholeskyDecompHelpers::_decomposerGenDim;
      fOk = _decomposerGenDim<F, M>()(fL, m, fN);
   }

   /// perform a Cholesky decomposition
   /** perfrom a Cholesky decomposition of a symmetric positive
    * definite matrix m
    *
    * this is the constructor to use in special applications where
    * plain arrays are used
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(i,j) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> CholeskyDecompGenDim(unsigned N, G* m) :
      fN(N), fL(new F[(fN * (fN + 1)) / 2]), fOk(false)
   {
      using CholeskyDecompHelpers::_decomposerGenDim;
      using CholeskyDecompHelpers::PackedArrayAdapter;
      fOk = _decomposerGenDim<F, PackedArrayAdapter<G> >()(
         fL, PackedArrayAdapter<G>(m), fN);
   }

   /// destructor
   ~CholeskyDecompGenDim() { delete[] fL; }

   /// returns true if decomposition was successful
   /** @returns true if decomposition was successful */
   bool ok() const { return fOk; }
   /// returns true if decomposition was successful
   /** @returns true if decomposition was successful */
   operator bool() const { return fOk; }

   /** @brief solves a linear system for the given right hand side
    *
    * Note that you can use both SVector classes and plain arrays for
    * rhs. (Make sure that the sizes match!). It will work with any vector
    * implementing the operator [i]
    *
    * @returns if the decomposition was successful
    */
   template<class V> bool Solve(V& rhs) const
   {
      using CholeskyDecompHelpers::_solverGenDim;
      if (fOk) _solverGenDim<F,V>()(rhs, fL, fN);
      return fOk;
   }

   /** @brief place the inverse into m
    *
    * This is the method to use with an SMatrix.
    *
    * @returns if the decomposition was successful
    */
   template<class M> bool Invert(M& m) const
   {
      using CholeskyDecompHelpers::_inverterGenDim;
      if (fOk) _inverterGenDim<F,M>()(m, fL, fN);
      return fOk;
   }

   /** @brief place the inverse into m
    *
    * This is the method to use with a plain array.
    *
    * @returns if the decomposition was successful
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(i,j) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> bool Invert(G* m) const
   {
      using CholeskyDecompHelpers::_inverterGenDim;
      using CholeskyDecompHelpers::PackedArrayAdapter;
      if (fOk) {
         PackedArrayAdapter<G> adapted(m);
         _inverterGenDim<F,PackedArrayAdapter<G> >()(adapted, fL, fN);
      }
      return fOk;
   }

   /** @brief obtain the decomposed matrix L
    *
    * This is the method to use with a plain array.
    *
    * @returns if the decomposition was successful
    */
   template<class M> bool getL(M& m) const
   {
      if (!fOk) return false;
      for (unsigned i = 0; i < fN; ++i) {
         // zero upper half of matrix
         for (unsigned j = i + 1; j < fN; ++j)
            m(i, j) = F(0);
         // copy the rest
         for (unsigned j = 0; j <= i; ++j)
            m(i, j) = fL[i * (i + 1) / 2 + j];
         // adjust the diagonal - we save 1/L(i, i) in that position, so
         // convert to what caller expects
         m(i, i) = F(1) / m(i, i);
      }
      return true;
   }

   /** @brief obtain the decomposed matrix L
    *
    * @returns if the decomposition was successful
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(i,j) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> bool getL(G* m) const
   {
       if (!fOk) return false;
       // copy L
       for (unsigned i = 0; i < (fN * (fN + 1)) / 2; ++i)
          m[i] = fL[i];
       // adjust diagonal - we save 1/L(i, i) in that position, so convert to
       // what caller expects
       for (unsigned i = 0; i < fN; ++i)
          m[(i * (i + 1)) / 2 + i] = F(1) / fL[(i * (i + 1)) / 2 + i];
       return true;
   }

   /** @brief obtain the inverse of the decomposed matrix L
    *
    * This is the method to use with a plain array.
    *
    * @returns if the decomposition was successful
    */
   template<class M> bool getLi(M& m) const
   {
      if (!fOk) return false;
      for (unsigned i = 0; i < fN; ++i) {
         // zero lower half of matrix
         for (unsigned j = i + 1; j < fN; ++j)
            m(j, i) = F(0);
         // copy the rest
         for (unsigned j = 0; j <= i; ++j)
            m(j, i) = fL[i * (i + 1) / 2 + j];
      }
      // invert the off-diagonal part of what we just copied
      for (unsigned i = 1; i < fN; ++i) {
         for (unsigned j = 0; j < i; ++j) {
            typename M::value_type tmp = F(0);
            for (unsigned k = i; k-- > j;)
               tmp -= m(k, i) * m(j, k);
            m(j, i) = tmp * m(i, i);
         }
      }
      return true;
   }

   /** @brief obtain the inverse of the decomposed matrix L
    *
    * @returns if the decomposition was successful
    *
    * NOTE: the matrix is given in packed representation, matrix
    * element m(j,i) (j <= i) is supposed to be in array element
    * (i * (i + 1)) / 2 + j
    */
   template<typename G> bool getLi(G* m) const
   {
       if (!fOk) return false;
      // copy L
      for (unsigned i = 0; i < (fN * (fN + 1)) / 2; ++i)
         m[i] = fL[i];
      // invert the off-diagonal part of what we just copied
      G* base1 = &m[1];
      for (unsigned i = 1; i < fN; base1 += ++i) {
         for (unsigned j = 0; j < i; ++j) {
            G tmp = F(0);
            const G *base2 = &m[(i * (i - 1)) / 2];
            for (unsigned k = i; k-- > j; base2 -= k)
              tmp -= base1[k] * base2[j];
            base1[j] = tmp * base1[i];
         }
      }
      return true;
   }
};

namespace CholeskyDecompHelpers {
   /// adapter for packed arrays (to SMatrix indexing conventions)
   template<typename G> class PackedArrayAdapter
   {
   private:
      G* fArr; ///< pointer to first array element
   public:
      /// constructor
      PackedArrayAdapter(G* arr) : fArr(arr) {}
      /// read access to elements (make sure that j <= i)
      const G operator()(unsigned i, unsigned j) const
      { return fArr[((i * (i + 1)) / 2) + j]; }
      /// write access to elements (make sure that j <= i)
      G& operator()(unsigned i, unsigned j)
      { return fArr[((i * (i + 1)) / 2) + j]; }
   };
   /// struct to do a Cholesky decomposition (general dimensionality)
   template<class F, class M> struct _decomposerGenDim
   {
      /// method to do the decomposition
      /** @returns if the decomposition was successful */
      bool operator()(F* dst, const M& src, unsigned N) const
      {
         // perform Cholesky decomposition of matrix: M = L L^T
         // only thing that can go wrong: trying to take square
         // root of negative number or zero (matrix is
         // ill-conditioned or singular in these cases)

         // element L(i,j) is at array position (i * (i+1)) / 2 + j

         // quirk: we may need to invert L later anyway, so we can
         // invert elements on diagonale straight away (we only
         // ever need their reciprocals!)

         // cache starting address of rows of L for speed reasons
         F *base1 = &dst[0];
         for (unsigned i = 0; i < N; base1 += ++i) {
            F tmpdiag = F(0.0); // for element on diagonale
            // calculate off-diagonal elements
            F *base2 = &dst[0];
            for (unsigned j = 0; j < i; base2 += ++j) {
               F tmp = src(i, j);
               for (unsigned k = j; k--; )
                  tmp -= base1[k] * base2[k];
               base1[j] = tmp *= base2[j];
               // keep track of contribution to element on diagonale
               tmpdiag += tmp * tmp;
            }
            // keep truncation error small
            tmpdiag = src(i, i) - tmpdiag;
            // check if positive definite
            if (tmpdiag <= F(0.0)) return false;
            else base1[i] = std::sqrt(F(1.0) / tmpdiag);
         }
         return true;
      }
   };

   /// struct to do a Cholesky decomposition
   template<class F, unsigned N, class M> struct _decomposer
   {
      /// method to do the decomposition
      /** @returns if the decomposition was successful */
      bool operator()(F* dst, const M& src) const
      { return _decomposerGenDim<F, M>()(dst, src, N); }
   };

   /// struct to obtain the inverse from a Cholesky decomposition (general dimensionality)
   template<class F, class M> struct _inverterGenDim
   {
      /// method to do the inversion
      void operator()(M& dst, const F* src, unsigned N) const
      {
         // make working copy
         F * l = new F[N * (N + 1) / 2];
         std::copy(src, src + ((N * (N + 1)) / 2), l);
         // ok, next step: invert off-diagonal part of matrix
         F* base1 = &l[1];
         for (unsigned i = 1; i < N; base1 += ++i) {
            for (unsigned j = 0; j < i; ++j) {
               F tmp = F(0.0);
               const F *base2 = &l[(i * (i - 1)) / 2];
               for (unsigned k = i; k-- > j; base2 -= k)
                  tmp -= base1[k] * base2[j];
               base1[j] = tmp * base1[i];
            }
         }

         // Li = L^(-1) formed, now calculate M^(-1) = Li^T Li
         for (unsigned i = N; i--; ) {
            for (unsigned j = i + 1; j--; ) {
               F tmp = F(0.0);
               base1 = &l[(N * (N - 1)) / 2];
               for (unsigned k = N; k-- > i; base1 -= k)
                  tmp += base1[i] * base1[j];
               dst(i, j) = tmp;
            }
         }
         delete [] l;
      }
   };

   /// struct to obtain the inverse from a Cholesky decomposition
   template<class F, unsigned N, class M> struct _inverter
   {
      /// method to do the inversion
      void operator()(M& dst, const F* src) const
      { return _inverterGenDim<F, M>()(dst, src, N); }
   };

   /// struct to solve a linear system using its Cholesky decomposition (generalised dimensionality)
   template<class F, class V> struct _solverGenDim
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l, unsigned N) const
      {
         // solve Ly = rhs
         for (unsigned k = 0; k < N; ++k) {
            const unsigned base = (k * (k + 1)) / 2;
            F sum = F(0.0);
            for (unsigned i = k; i--; )
               sum += rhs[i] * l[base + i];
            // elements on diagonale are pre-inverted!
            rhs[k] = (rhs[k] - sum) * l[base + k];
         }
         // solve L^Tx = y
         for (unsigned k = N; k--; ) {
            F sum = F(0.0);
            for (unsigned i = N; --i > k; )
               sum += rhs[i] * l[(i * (i + 1)) / 2 + k];
            // elements on diagonale are pre-inverted!
            rhs[k] = (rhs[k] - sum) * l[(k * (k + 1)) / 2 + k];
         }
      }
   };

   /// struct to solve a linear system using its Cholesky decomposition
   template<class F, unsigned N, class V> struct _solver
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      { _solverGenDim<F, V>()(rhs, l, N); }
   };

   /// struct to do a Cholesky decomposition (specialized, N = 6)
   template<class F, class M> struct _decomposer<F, 6, M>
   {
      /// method to do the decomposition
      bool operator()(F* dst, const M& src) const
      {
         if (src(0,0) <= F(0.0)) return false;
         dst[0] = std::sqrt(F(1.0) / src(0,0));
         dst[1] = src(1,0) * dst[0];
         dst[2] = src(1,1) - dst[1] * dst[1];
         if (dst[2] <= F(0.0)) return false;
         else dst[2] = std::sqrt(F(1.0) / dst[2]);
         dst[3] = src(2,0) * dst[0];
         dst[4] = (src(2,1) - dst[1] * dst[3]) * dst[2];
         dst[5] = src(2,2) - (dst[3] * dst[3] + dst[4] * dst[4]);
         if (dst[5] <= F(0.0)) return false;
         else dst[5] = std::sqrt(F(1.0) / dst[5]);
         dst[6] = src(3,0) * dst[0];
         dst[7] = (src(3,1) - dst[1] * dst[6]) * dst[2];
         dst[8] = (src(3,2) - dst[3] * dst[6] - dst[4] * dst[7]) * dst[5];
         dst[9] = src(3,3) - (dst[6] * dst[6] + dst[7] * dst[7] + dst[8] * dst[8]);
         if (dst[9] <= F(0.0)) return false;
         else dst[9] = std::sqrt(F(1.0) / dst[9]);
         dst[10] = src(4,0) * dst[0];
         dst[11] = (src(4,1) - dst[1] * dst[10]) * dst[2];
         dst[12] = (src(4,2) - dst[3] * dst[10] - dst[4] * dst[11]) * dst[5];
         dst[13] = (src(4,3) - dst[6] * dst[10] - dst[7] * dst[11] - dst[8] * dst[12]) * dst[9];
         dst[14] = src(4,4) - (dst[10]*dst[10]+dst[11]*dst[11]+dst[12]*dst[12]+dst[13]*dst[13]);
         if (dst[14] <= F(0.0)) return false;
         else dst[14] = std::sqrt(F(1.0) / dst[14]);
         dst[15] = src(5,0) * dst[0];
         dst[16] = (src(5,1) - dst[1] * dst[15]) * dst[2];
         dst[17] = (src(5,2) - dst[3] * dst[15] - dst[4] * dst[16]) * dst[5];
         dst[18] = (src(5,3) - dst[6] * dst[15] - dst[7] * dst[16] - dst[8] * dst[17]) * dst[9];
         dst[19] = (src(5,4) - dst[10] * dst[15] - dst[11] * dst[16] - dst[12] * dst[17] - dst[13] * dst[18]) * dst[14];
         dst[20] = src(5,5) - (dst[15]*dst[15]+dst[16]*dst[16]+dst[17]*dst[17]+dst[18]*dst[18]+dst[19]*dst[19]);
         if (dst[20] <= F(0.0)) return false;
         else dst[20] = std::sqrt(F(1.0) / dst[20]);
         return true;
      }
   };
   /// struct to do a Cholesky decomposition (specialized, N = 5)
   template<class F, class M> struct _decomposer<F, 5, M>
   {
      /// method to do the decomposition
      bool operator()(F* dst, const M& src) const
      {
         if (src(0,0) <= F(0.0)) return false;
         dst[0] = std::sqrt(F(1.0) / src(0,0));
         dst[1] = src(1,0) * dst[0];
         dst[2] = src(1,1) - dst[1] * dst[1];
         if (dst[2] <= F(0.0)) return false;
         else dst[2] = std::sqrt(F(1.0) / dst[2]);
         dst[3] = src(2,0) * dst[0];
         dst[4] = (src(2,1) - dst[1] * dst[3]) * dst[2];
         dst[5] = src(2,2) - (dst[3] * dst[3] + dst[4] * dst[4]);
         if (dst[5] <= F(0.0)) return false;
         else dst[5] = std::sqrt(F(1.0) / dst[5]);
         dst[6] = src(3,0) * dst[0];
         dst[7] = (src(3,1) - dst[1] * dst[6]) * dst[2];
         dst[8] = (src(3,2) - dst[3] * dst[6] - dst[4] * dst[7]) * dst[5];
         dst[9] = src(3,3) - (dst[6] * dst[6] + dst[7] * dst[7] + dst[8] * dst[8]);
         if (dst[9] <= F(0.0)) return false;
         else dst[9] = std::sqrt(F(1.0) / dst[9]);
         dst[10] = src(4,0) * dst[0];
         dst[11] = (src(4,1) - dst[1] * dst[10]) * dst[2];
         dst[12] = (src(4,2) - dst[3] * dst[10] - dst[4] * dst[11]) * dst[5];
         dst[13] = (src(4,3) - dst[6] * dst[10] - dst[7] * dst[11] - dst[8] * dst[12]) * dst[9];
         dst[14] = src(4,4) - (dst[10]*dst[10]+dst[11]*dst[11]+dst[12]*dst[12]+dst[13]*dst[13]);
         if (dst[14] <= F(0.0)) return false;
         else dst[14] = std::sqrt(F(1.0) / dst[14]);
         return true;
      }
   };
   /// struct to do a Cholesky decomposition (specialized, N = 4)
   template<class F, class M> struct _decomposer<F, 4, M>
   {
      /// method to do the decomposition
      bool operator()(F* dst, const M& src) const
      {
         if (src(0,0) <= F(0.0)) return false;
         dst[0] = std::sqrt(F(1.0) / src(0,0));
         dst[1] = src(1,0) * dst[0];
         dst[2] = src(1,1) - dst[1] * dst[1];
         if (dst[2] <= F(0.0)) return false;
         else dst[2] = std::sqrt(F(1.0) / dst[2]);
         dst[3] = src(2,0) * dst[0];
         dst[4] = (src(2,1) - dst[1] * dst[3]) * dst[2];
         dst[5] = src(2,2) - (dst[3] * dst[3] + dst[4] * dst[4]);
         if (dst[5] <= F(0.0)) return false;
         else dst[5] = std::sqrt(F(1.0) / dst[5]);
         dst[6] = src(3,0) * dst[0];
         dst[7] = (src(3,1) - dst[1] * dst[6]) * dst[2];
         dst[8] = (src(3,2) - dst[3] * dst[6] - dst[4] * dst[7]) * dst[5];
         dst[9] = src(3,3) - (dst[6] * dst[6] + dst[7] * dst[7] + dst[8] * dst[8]);
         if (dst[9] <= F(0.0)) return false;
         else dst[9] = std::sqrt(F(1.0) / dst[9]);
         return true;
      }
   };
   /// struct to do a Cholesky decomposition (specialized, N = 3)
   template<class F, class M> struct _decomposer<F, 3, M>
   {
      /// method to do the decomposition
      bool operator()(F* dst, const M& src) const
      {
         if (src(0,0) <= F(0.0)) return false;
         dst[0] = std::sqrt(F(1.0) / src(0,0));
         dst[1] = src(1,0) * dst[0];
         dst[2] = src(1,1) - dst[1] * dst[1];
         if (dst[2] <= F(0.0)) return false;
         else dst[2] = std::sqrt(F(1.0) / dst[2]);
         dst[3] = src(2,0) * dst[0];
         dst[4] = (src(2,1) - dst[1] * dst[3]) * dst[2];
         dst[5] = src(2,2) - (dst[3] * dst[3] + dst[4] * dst[4]);
         if (dst[5] <= F(0.0)) return false;
         else dst[5] = std::sqrt(F(1.0) / dst[5]);
         return true;
      }
   };
   /// struct to do a Cholesky decomposition (specialized, N = 2)
   template<class F, class M> struct _decomposer<F, 2, M>
   {
      /// method to do the decomposition
      bool operator()(F* dst, const M& src) const
      {
         if (src(0,0) <= F(0.0)) return false;
         dst[0] = std::sqrt(F(1.0) / src(0,0));
         dst[1] = src(1,0) * dst[0];
         dst[2] = src(1,1) - dst[1] * dst[1];
         if (dst[2] <= F(0.0)) return false;
         else dst[2] = std::sqrt(F(1.0) / dst[2]);
         return true;
      }
   };
   /// struct to do a Cholesky decomposition (specialized, N = 1)
   template<class F, class M> struct _decomposer<F, 1, M>
   {
      /// method to do the decomposition
      bool operator()(F* dst, const M& src) const
      {
         if (src(0,0) <= F(0.0)) return false;
         dst[0] = std::sqrt(F(1.0) / src(0,0));
         return true;
      }
   };
   /// struct to do a Cholesky decomposition (specialized, N = 0)
   template<class F, class M> struct _decomposer<F, 0, M>
   {
   private:
      _decomposer() { };
      bool operator()(F* dst, const M& src) const;
   };

   /// struct to obtain the inverse from a Cholesky decomposition (N = 6)
   template<class F, class M> struct _inverter<F,6,M>
   {
      /// method to do the inversion
      inline void operator()(M& dst, const F* src) const
      {
         const F li21 = -src[1] * src[0] * src[2];
         const F li32 = -src[4] * src[2] * src[5];
         const F li31 = (src[1] * src[4] * src[2] - src[3]) * src[0] * src[5];
         const F li43 = -src[8] * src[9] * src[5];
         const F li42 = (src[4] * src[8] * src[5] - src[7]) * src[2] * src[9];
         const F li41 = (-src[1] * src[4] * src[8] * src[2] * src[5] +
                         src[1] * src[7] * src[2] + src[3] * src[8] * src[5] - src[6]) * src[0] * src[9];
         const F li54 = -src[13] * src[14] * src[9];
         const F li53 = (src[13] * src[8] * src[9] - src[12]) * src[5] * src[14];
         const F li52 = (-src[4] * src[8] * src[13] * src[5] * src[9] +
                         src[4] * src[12] * src[5] + src[7] * src[13] * src[9] - src[11]) * src[2] * src[14];
         const F li51 = (src[1]*src[4]*src[8]*src[13]*src[2]*src[5]*src[9] -
                         src[13]*src[8]*src[3]*src[9]*src[5] - src[12]*src[4]*src[1]*src[2]*src[5] - src[13]*src[7]*src[1]*src[9]*src[2] +
                         src[11]*src[1]*src[2] + src[12]*src[3]*src[5] + src[13]*src[6]*src[9] -src[10]) * src[0] * src[14];
         const F li65 = -src[19] * src[20] * src[14];
         const F li64 = (src[19] * src[13] * src[14] - src[18]) * src[9] * src[20];
         const F li63 = (-src[8] * src[13] * src[19] * src[9] * src[14] +
                         src[8] * src[18] * src[9] + src[12] * src[19] * src[14] - src[17]) * src[5] * src[20];
         const F li62 = (src[4]*src[8]*src[13]*src[19]*src[5]*src[9]*src[14] -
                         src[18]*src[8]*src[4]*src[9]*src[5] - src[19]*src[12]*src[4]*src[14]*src[5] -src[19]*src[13]*src[7]*src[14]*src[9] +
                         src[17]*src[4]*src[5] + src[18]*src[7]*src[9] + src[19]*src[11]*src[14] - src[16]) * src[2] * src[20];
         const F li61 = (-src[19]*src[13]*src[8]*src[4]*src[1]*src[2]*src[5]*src[9]*src[14] +
                         src[18]*src[8]*src[4]*src[1]*src[2]*src[5]*src[9] + src[19]*src[12]*src[4]*src[1]*src[2]*src[5]*src[14] +
                         src[19]*src[13]*src[7]*src[1]*src[2]*src[9]*src[14] + src[19]*src[13]*src[8]*src[3]*src[5]*src[9]*src[14] -
                         src[17]*src[4]*src[1]*src[2]*src[5] - src[18]*src[7]*src[1]*src[2]*src[9] - src[19]*src[11]*src[1]*src[2]*src[14] -
                         src[18]*src[8]*src[3]*src[5]*src[9] - src[19]*src[12]*src[3]*src[5]*src[14] - src[19]*src[13]*src[6]*src[9]*src[14] +
                         src[16]*src[1]*src[2] + src[17]*src[3]*src[5] + src[18]*src[6]*src[9] + src[19]*src[10]*src[14] - src[15]) *
            src[0] * src[20];

         dst(0,0) = li61*li61 + li51*li51 + li41*li41 + li31*li31 + li21*li21 + src[0]*src[0];
         dst(1,0) = li61*li62 + li51*li52 + li41*li42 + li31*li32 + li21*src[2];
         dst(1,1) = li62*li62 + li52*li52 + li42*li42 + li32*li32 + src[2]*src[2];
         dst(2,0) = li61*li63 + li51*li53 + li41*li43 + li31*src[5];
         dst(2,1) = li62*li63 + li52*li53 + li42*li43 + li32*src[5];
         dst(2,2) = li63*li63 + li53*li53 + li43*li43 + src[5]*src[5];
         dst(3,0) = li61*li64 + li51*li54 + li41*src[9];
         dst(3,1) = li62*li64 + li52*li54 + li42*src[9];
         dst(3,2) = li63*li64 + li53*li54 + li43*src[9];
         dst(3,3) = li64*li64 + li54*li54 + src[9]*src[9];
         dst(4,0) = li61*li65 + li51*src[14];
         dst(4,1) = li62*li65 + li52*src[14];
         dst(4,2) = li63*li65 + li53*src[14];
         dst(4,3) = li64*li65 + li54*src[14];
         dst(4,4) = li65*li65 + src[14]*src[14];
         dst(5,0) = li61*src[20];
         dst(5,1) = li62*src[20];
         dst(5,2) = li63*src[20];
         dst(5,3) = li64*src[20];
         dst(5,4) = li65*src[20];
         dst(5,5) = src[20]*src[20];
      }
   };
   /// struct to obtain the inverse from a Cholesky decomposition (N = 5)
   template<class F, class M> struct _inverter<F,5,M>
   {
      /// method to do the inversion
      inline void operator()(M& dst, const F* src) const
      {
         const F li21 = -src[1] * src[0] * src[2];
         const F li32 = -src[4] * src[2] * src[5];
         const F li31 = (src[1] * src[4] * src[2] - src[3]) * src[0] * src[5];
         const F li43 = -src[8] * src[9] * src[5];
         const F li42 = (src[4] * src[8] * src[5] - src[7]) * src[2] * src[9];
         const F li41 = (-src[1] * src[4] * src[8] * src[2] * src[5] +
                         src[1] * src[7] * src[2] + src[3] * src[8] * src[5] - src[6]) * src[0] * src[9];
         const F li54 = -src[13] * src[14] * src[9];
         const F li53 = (src[13] * src[8] * src[9] - src[12]) * src[5] * src[14];
         const F li52 = (-src[4] * src[8] * src[13] * src[5] * src[9] +
                         src[4] * src[12] * src[5] + src[7] * src[13] * src[9] - src[11]) * src[2] * src[14];
         const F li51 = (src[1]*src[4]*src[8]*src[13]*src[2]*src[5]*src[9] -
                         src[13]*src[8]*src[3]*src[9]*src[5] - src[12]*src[4]*src[1]*src[2]*src[5] - src[13]*src[7]*src[1]*src[9]*src[2] +
                         src[11]*src[1]*src[2] + src[12]*src[3]*src[5] + src[13]*src[6]*src[9] -src[10]) * src[0] * src[14];

         dst(0,0) = li51*li51 + li41*li41 + li31*li31 + li21*li21 + src[0]*src[0];
         dst(1,0) = li51*li52 + li41*li42 + li31*li32 + li21*src[2];
         dst(1,1) = li52*li52 + li42*li42 + li32*li32 + src[2]*src[2];
         dst(2,0) = li51*li53 + li41*li43 + li31*src[5];
         dst(2,1) = li52*li53 + li42*li43 + li32*src[5];
         dst(2,2) = li53*li53 + li43*li43 + src[5]*src[5];
         dst(3,0) = li51*li54 + li41*src[9];
         dst(3,1) = li52*li54 + li42*src[9];
         dst(3,2) = li53*li54 + li43*src[9];
         dst(3,3) = li54*li54 + src[9]*src[9];
         dst(4,0) = li51*src[14];
         dst(4,1) = li52*src[14];
         dst(4,2) = li53*src[14];
         dst(4,3) = li54*src[14];
         dst(4,4) = src[14]*src[14];
      }
   };
   /// struct to obtain the inverse from a Cholesky decomposition (N = 4)
   template<class F, class M> struct _inverter<F,4,M>
   {
      /// method to do the inversion
      inline void operator()(M& dst, const F* src) const
      {
         const F li21 = -src[1] * src[0] * src[2];
         const F li32 = -src[4] * src[2] * src[5];
         const F li31 = (src[1] * src[4] * src[2] - src[3]) * src[0] * src[5];
         const F li43 = -src[8] * src[9] * src[5];
         const F li42 = (src[4] * src[8] * src[5] - src[7]) * src[2] * src[9];
         const F li41 = (-src[1] * src[4] * src[8] * src[2] * src[5] +
                         src[1] * src[7] * src[2] + src[3] * src[8] * src[5] - src[6]) * src[0] * src[9];

         dst(0,0) = li41*li41 + li31*li31 + li21*li21 + src[0]*src[0];
         dst(1,0) = li41*li42 + li31*li32 + li21*src[2];
         dst(1,1) = li42*li42 + li32*li32 + src[2]*src[2];
         dst(2,0) = li41*li43 + li31*src[5];
         dst(2,1) = li42*li43 + li32*src[5];
         dst(2,2) = li43*li43 + src[5]*src[5];
         dst(3,0) = li41*src[9];
         dst(3,1) = li42*src[9];
         dst(3,2) = li43*src[9];
         dst(3,3) = src[9]*src[9];
      }
   };
   /// struct to obtain the inverse from a Cholesky decomposition (N = 3)
   template<class F, class M> struct _inverter<F,3,M>
   {
      /// method to do the inversion
      inline void operator()(M& dst, const F* src) const
      {
         const F li21 = -src[1] * src[0] * src[2];
         const F li32 = -src[4] * src[2] * src[5];
         const F li31 = (src[1] * src[4] * src[2] - src[3]) * src[0] * src[5];

         dst(0,0) = li31*li31 + li21*li21 + src[0]*src[0];
         dst(1,0) = li31*li32 + li21*src[2];
         dst(1,1) = li32*li32 + src[2]*src[2];
         dst(2,0) = li31*src[5];
         dst(2,1) = li32*src[5];
         dst(2,2) = src[5]*src[5];
      }
   };
   /// struct to obtain the inverse from a Cholesky decomposition (N = 2)
   template<class F, class M> struct _inverter<F,2,M>
   {
      /// method to do the inversion
      inline void operator()(M& dst, const F* src) const
      {
         const F li21 = -src[1] * src[0] * src[2];

         dst(0,0) = li21*li21 + src[0]*src[0];
         dst(1,0) = li21*src[2];
         dst(1,1) = src[2]*src[2];
      }
   };
   /// struct to obtain the inverse from a Cholesky decomposition (N = 1)
   template<class F, class M> struct _inverter<F,1,M>
   {
      /// method to do the inversion
      inline void operator()(M& dst, const F* src) const
      {
         dst(0,0) = src[0]*src[0];
      }
   };
   /// struct to obtain the inverse from a Cholesky decomposition (N = 0)
   template<class F, class M> struct _inverter<F,0,M>
   {
   private:
      _inverter();
      void operator()(M& dst, const F* src) const;
   };

   /// struct to solve a linear system using its Cholesky decomposition (N=6)
   template<class F, class V> struct _solver<F,6,V>
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      {
         // solve Ly = rhs
         const F y0 = rhs[0] * l[0];
         const F y1 = (rhs[1]-l[1]*y0)*l[2];
         const F y2 = (rhs[2]-(l[3]*y0+l[4]*y1))*l[5];
         const F y3 = (rhs[3]-(l[6]*y0+l[7]*y1+l[8]*y2))*l[9];
         const F y4 = (rhs[4]-(l[10]*y0+l[11]*y1+l[12]*y2+l[13]*y3))*l[14];
         const F y5 = (rhs[5]-(l[15]*y0+l[16]*y1+l[17]*y2+l[18]*y3+l[19]*y4))*l[20];
         // solve L^Tx = y, and put x into rhs
         rhs[5] = y5 * l[20];
         rhs[4] = (y4-l[19]*rhs[5])*l[14];
         rhs[3] = (y3-(l[18]*rhs[5]+l[13]*rhs[4]))*l[9];
         rhs[2] = (y2-(l[17]*rhs[5]+l[12]*rhs[4]+l[8]*rhs[3]))*l[5];
         rhs[1] = (y1-(l[16]*rhs[5]+l[11]*rhs[4]+l[7]*rhs[3]+l[4]*rhs[2]))*l[2];
         rhs[0] = (y0-(l[15]*rhs[5]+l[10]*rhs[4]+l[6]*rhs[3]+l[3]*rhs[2]+l[1]*rhs[1]))*l[0];
      }
   };
   /// struct to solve a linear system using its Cholesky decomposition (N=5)
   template<class F, class V> struct _solver<F,5,V>
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      {
         // solve Ly = rhs
         const F y0 = rhs[0] * l[0];
         const F y1 = (rhs[1]-l[1]*y0)*l[2];
         const F y2 = (rhs[2]-(l[3]*y0+l[4]*y1))*l[5];
         const F y3 = (rhs[3]-(l[6]*y0+l[7]*y1+l[8]*y2))*l[9];
         const F y4 = (rhs[4]-(l[10]*y0+l[11]*y1+l[12]*y2+l[13]*y3))*l[14];
         // solve L^Tx = y, and put x into rhs
         rhs[4] = (y4)*l[14];
         rhs[3] = (y3-(l[13]*rhs[4]))*l[9];
         rhs[2] = (y2-(l[12]*rhs[4]+l[8]*rhs[3]))*l[5];
         rhs[1] = (y1-(l[11]*rhs[4]+l[7]*rhs[3]+l[4]*rhs[2]))*l[2];
         rhs[0] = (y0-(l[10]*rhs[4]+l[6]*rhs[3]+l[3]*rhs[2]+l[1]*rhs[1]))*l[0];
      }
   };
   /// struct to solve a linear system using its Cholesky decomposition (N=4)
   template<class F, class V> struct _solver<F,4,V>
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      {
         // solve Ly = rhs
         const F y0 = rhs[0] * l[0];
         const F y1 = (rhs[1]-l[1]*y0)*l[2];
         const F y2 = (rhs[2]-(l[3]*y0+l[4]*y1))*l[5];
         const F y3 = (rhs[3]-(l[6]*y0+l[7]*y1+l[8]*y2))*l[9];
         // solve L^Tx = y, and put x into rhs
         rhs[3] = (y3)*l[9];
         rhs[2] = (y2-(l[8]*rhs[3]))*l[5];
         rhs[1] = (y1-(l[7]*rhs[3]+l[4]*rhs[2]))*l[2];
         rhs[0] = (y0-(l[6]*rhs[3]+l[3]*rhs[2]+l[1]*rhs[1]))*l[0];
      }
   };
   /// struct to solve a linear system using its Cholesky decomposition (N=3)
   template<class F, class V> struct _solver<F,3,V>
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      {
         // solve Ly = rhs
         const F y0 = rhs[0] * l[0];
         const F y1 = (rhs[1]-l[1]*y0)*l[2];
         const F y2 = (rhs[2]-(l[3]*y0+l[4]*y1))*l[5];
         // solve L^Tx = y, and put x into rhs
         rhs[2] = (y2)*l[5];
         rhs[1] = (y1-(l[4]*rhs[2]))*l[2];
         rhs[0] = (y0-(l[3]*rhs[2]+l[1]*rhs[1]))*l[0];
      }
   };
   /// struct to solve a linear system using its Cholesky decomposition (N=2)
   template<class F, class V> struct _solver<F,2,V>
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      {
         // solve Ly = rhs
         const F y0 = rhs[0] * l[0];
         const F y1 = (rhs[1]-l[1]*y0)*l[2];
         // solve L^Tx = y, and put x into rhs
         rhs[1] = (y1)*l[2];
         rhs[0] = (y0-(l[1]*rhs[1]))*l[0];
      }
   };
   /// struct to solve a linear system using its Cholesky decomposition (N=1)
   template<class F, class V> struct _solver<F,1,V>
   {
      /// method to solve the linear system
      void operator()(V& rhs, const F* l) const
      {
         // solve LL^T x = rhs, put y into rhs
         rhs[0] *= l[0] * l[0];
      }
   };
   /// struct to solve a linear system using its Cholesky decomposition (N=0)
   template<class F, class V> struct _solver<F,0,V>
   {
   private:
      _solver();
      void operator()(V& rhs, const F* l) const;
   };
}


   }  // namespace Math

}  // namespace ROOT

#endif // ROOT_Math_CHOLESKYDECOMP

