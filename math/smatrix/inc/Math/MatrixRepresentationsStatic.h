// @(#)root/smatrix:$Id$
// Author: L. Moneta, J. Palacios    2006  

#ifndef ROOT_Math_MatrixRepresentationsStatic
#define ROOT_Math_MatrixRepresentationsStatic 1

//#ifndef SMATRIX_USE_COMPUTATION
//#define SMATRIX_USE_CONSTEXPR
//#endif


#ifndef SMATRIX_USE_CONSTEXPR
#define SMATRIX_USE_COMPUTATION
#endif


// Include files

/** 
    @defgroup MatRep SMatrix Storage Representation 
    @ingroup SMatrixGroup
 
    @author Juan Palacios
    @date   2006-01-15
 
    Classes MatRepStd and MatRepSym for generic and symmetric matrix
    data storage and manipulation. Define data storage and access, plus
    operators =, +=, -=, ==.
 
 */

#ifndef ROOT_Math_StaticCheck
#include "Math/StaticCheck.h"
#endif

#ifdef SMATRIX_USE_CONSTEXPR
#include <cstddef>
#include <utility>
#include <type_traits>
#include <array>
#endif


namespace ROOT {
   
namespace Math {

   //________________________________________________________________________________    
   /**
      MatRepStd
      Standard Matrix representation for a general D1 x D2 matrix. 
      This class is itself a template on the contained type T, the number of rows and the number of columns.
      Its data member is an array T[nrows*ncols] containing the matrix data. 
      The data are stored in the row-major C convention. 
      For example, for a matrix, M, of size 3x3, the data \f$ \left[a_0,a_1,a_2,.......,a_7,a_8 \right] \f$d are stored in the following order: 
      \f[
      M = \left( \begin{array}{ccc} 
      a_0 & a_1 & a_2  \\ 
      a_3 & a_4  & a_5  \\ 
      a_6 & a_7  & a_8   \end{array} \right)
      \f]

      @ingroup MatRep
   */


   template <class T, unsigned int D1, unsigned int D2=D1>
   class MatRepStd {

   public: 

      typedef T  value_type;

      inline const T& operator()(unsigned int i, unsigned int j) const {
         return fArray[i*D2+j];
      }
      inline T& operator()(unsigned int i, unsigned int j) {
         return fArray[i*D2+j];
      }
      inline T& operator[](unsigned int i) { return fArray[i]; }

      inline const T& operator[](unsigned int i) const { return fArray[i]; }

      inline T apply(unsigned int i) const { return fArray[i]; }

      inline T* Array() { return fArray; }  

      inline const T* Array() const { return fArray; }  

      template <class R>
      inline MatRepStd<T, D1, D2>& operator+=(const R& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] += rhs[i];
         return *this;
      }

      template <class R>
      inline MatRepStd<T, D1, D2>& operator-=(const R& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] -= rhs[i];
         return *this;
      }

      template <class R>
      inline MatRepStd<T, D1, D2>& operator=(const R& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs[i];
         return *this;
      }

      template <class R> 
      inline bool operator==(const R& rhs) const {
         bool rc = true;
         for(unsigned int i=0; i<kSize; ++i) {
            rc = rc && (fArray[i] == rhs[i]);
         }
         return rc;
      }

      enum {
         /// return no. of matrix rows
         kRows = D1,
         /// return no. of matrix columns
         kCols = D2,
         /// return no of elements: rows*columns
         kSize = D1*D2
      };
      
   private:
      //T __attribute__ ((aligned (16))) fArray[kSize];
      T  fArray[kSize];
   };
    
    
//     template<unigned int D>
//     struct Creator { 
//       static const RowOffsets<D> & Offsets() {
// 	static RowOffsets<D> off;
// 	return off;
//       }

   /**
      Static structure to keep the conversion from (i,j) to offsets in the storage data for a 
      symmetric matrix
   */

   template<unsigned int D>
   struct RowOffsets {
      inline RowOffsets() {
         int v[D];
         v[0]=0;
         for (unsigned int i=1; i<D; ++i)
            v[i]=v[i-1]+i;
         for (unsigned int i=0; i<D; ++i) { 
            for (unsigned int j=0; j<=i; ++j)
               fOff[i*D+j] = v[i]+j; 
            for (unsigned int j=i+1; j<D; ++j)
               fOff[i*D+j] = v[j]+i ;
         }
      }
      inline int operator()(unsigned int i, unsigned int j) const { return fOff[i*D+j]; }
      inline int apply(unsigned int i) const { return fOff[i]; }
      int fOff[D*D];
   };

#ifdef SMATRIX_USE_CONSTEXPR
  namespace rowOffsetsUtils {

    ///////////
    // Some meta template stuff
    template<int...> struct indices{};

    template<int I, class IndexTuple, int N>
    struct make_indices_impl;

    template<int I, int... Indices, int N>
    struct make_indices_impl<I, indices<Indices...>, N>
    {
      typedef typename make_indices_impl<I + 1, indices<Indices..., I>,
					 N>::type type;
    };

    template<int N, int... Indices>
    struct make_indices_impl<N, indices<Indices...>, N> {
      typedef indices<Indices...> type;
    };

    template<int N>
    struct make_indices : make_indices_impl<0, indices<>, N> {};
    // end of stuff



    template<int I0, class F, int... I>
    constexpr std::array<decltype(std::declval<F>()(std::declval<int>())), sizeof...(I)>
    do_make(F f, indices<I...>)
    {
      return  std::array<decltype(std::declval<F>()(std::declval<int>())),
			 sizeof...(I)>{{ f(I0 + I)... }};
    }

    template<int N, int I0 = 0, class F>
    constexpr std::array<decltype(std::declval<F>()(std::declval<int>())), N>
    make(F f) {
      return do_make<I0>(f, typename make_indices<N>::type());
    }

  } // namespace rowOffsetsUtils 

#elif defined(SMATRIX_USE_COMPUTATION)

#else



// Make the lookup tables available at compile time:
// Add them to a namespace?
static const int fOff1x1[] = {0};
static const int fOff2x2[] = {0, 1, 1, 2};
static const int fOff3x3[] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
static const int fOff4x4[] = {0, 1, 3, 6, 1, 2, 4, 7, 3, 4, 5, 8, 6, 7, 8, 9};
static const int fOff5x5[] = {0, 1, 3, 6, 10, 1, 2, 4, 7, 11, 3, 4, 5, 8, 12, 6, 7, 8, 9, 13, 10, 11, 12, 13, 14};
static const int fOff6x6[] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};

static const int fOff7x7[] = {0, 1, 3, 6, 10, 15, 21, 1, 2, 4, 7, 11, 16, 22, 3, 4, 5, 8, 12, 17, 23, 6, 7, 8, 9, 13, 18, 24, 10, 11, 12, 13, 14, 19, 25, 15, 16, 17, 18, 19, 20, 26, 21, 22, 23, 24, 25, 26, 27};

static const int fOff8x8[] = {0, 1, 3, 6, 10, 15, 21, 28, 1, 2, 4, 7, 11, 16, 22, 29, 3, 4, 5, 8, 12, 17, 23, 30, 6, 7, 8, 9, 13, 18, 24, 31, 10, 11, 12, 13, 14, 19, 25, 32, 15, 16, 17, 18, 19, 20, 26, 33, 21, 22, 23, 24, 25, 26, 27, 34, 28, 29, 30, 31, 32, 33, 34, 35};

static const int fOff9x9[] = {0, 1, 3, 6, 10, 15, 21, 28, 36, 1, 2, 4, 7, 11, 16, 22, 29, 37, 3, 4, 5, 8, 12, 17, 23, 30, 38, 6, 7, 8, 9, 13, 18, 24, 31, 39, 10, 11, 12, 13, 14, 19, 25, 32, 40, 15, 16, 17, 18, 19, 20, 26, 33, 41, 21, 22, 23, 24, 25, 26, 27, 34, 42, 28, 29, 30, 31, 32, 33, 34, 35, 43, 36, 37, 38, 39, 40, 41, 42, 43, 44};

static const int fOff10x10[] = {0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 3, 4, 5, 8, 12, 17, 23, 30, 38, 47, 6, 7, 8, 9, 13, 18, 24, 31, 39, 48, 10, 11, 12, 13, 14, 19, 25, 32, 40, 49, 15, 16, 17, 18, 19, 20, 26, 33, 41, 50, 21, 22, 23, 24, 25, 26, 27, 34, 42, 51, 28, 29, 30, 31, 32, 33, 34, 35, 43, 52, 36, 37, 38, 39, 40, 41, 42, 43, 44, 53, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};

template<>
	struct RowOffsets<1> {
	  RowOffsets() {}
	  int operator()(unsigned int , unsigned int ) const { return 0; } // Just one element
	  int apply(unsigned int ) const { return 0; }
	};

template<>
	struct RowOffsets<2> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return i+j; /*fOff2x2[i*2+j];*/ }
	  int apply(unsigned int i) const { return fOff2x2[i]; }
	};

template<>
	struct RowOffsets<3> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff3x3[i*3+j]; }
	  int apply(unsigned int i) const { return fOff3x3[i]; }
	};

template<>
	struct RowOffsets<4> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff4x4[i*4+j]; }
	  int apply(unsigned int i) const { return fOff4x4[i]; }
	};

	template<>
	struct RowOffsets<5> {
	  inline RowOffsets() {}
	  inline int operator()(unsigned int i, unsigned int j) const { return fOff5x5[i*5+j]; }
//	int operator()(unsigned int i, unsigned int j) const {
//	  if(j <= i) return (i * (i + 1)) / 2 + j;
//		else return (j * (j + 1)) / 2 + i;
//	  }  
	inline int apply(unsigned int i) const { return fOff5x5[i]; }
	};

template<>
	struct RowOffsets<6> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff6x6[i*6+j]; }
	  int apply(unsigned int i) const { return fOff6x6[i]; }
	};

template<>
	struct RowOffsets<7> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff7x7[i*7+j]; }
	  int apply(unsigned int i) const { return fOff7x7[i]; }
	};

template<>
	struct RowOffsets<8> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff8x8[i*8+j]; }
	  int apply(unsigned int i) const { return fOff8x8[i]; }
	};

template<>
	struct RowOffsets<9> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff9x9[i*9+j]; }
	  int apply(unsigned int i) const { return fOff9x9[i]; }
	};

template<>
	struct RowOffsets<10> {
	  RowOffsets() {}
	  int operator()(unsigned int i, unsigned int j) const { return fOff10x10[i*10+j]; }
	  int apply(unsigned int i) const { return fOff10x10[i]; }
	};

#endif

//_________________________________________________________________________________
   /**
      MatRepSym
      Matrix storage representation for a symmetric matrix of dimension NxN
      This class is a template on the contained type and on the symmetric matrix size, N. 
      It has as data member an array of type T of size N*(N+1)/2, 
      containing the lower diagonal block of the matrix.
      The order follows the lower diagonal block, still in a row-major convention. 
      For example for a symmetric 3x3 matrix the order of the 6 elements 
      \f$ \left[a_0,a_1.....a_5 \right]\f$ is: 
      \f[
      M = \left( \begin{array}{ccc} 
      a_0 & a_1  & a_3  \\ 
      a_1 & a_2  & a_4  \\
      a_3 & a_4 & a_5   \end{array} \right)
      \f]

      @ingroup MatRep 
   */
   template <class T, unsigned int D>
   class MatRepSym {

   public: 

#if defined(SMATRIX_USE_CONSTEXPR) || defined(SMATRIX_USE_COMPUTATION)
     /* constexpr */ inline MatRepSym(){}
#else
      MatRepSym() :fOff(0) { CreateOffsets(); } 
#endif
      typedef T  value_type;


    inline T & operator()(unsigned int i, unsigned int j)
     { return fArray[offset(i, j)]; }

     inline /* constexpr */ T const & operator()(unsigned int i, unsigned int j) const
     { return fArray[offset(i, j)]; }

     inline T& operator[](unsigned int i) { 
       return fArray[off(i)];
     }

     inline /* constexpr */ T const & operator[](unsigned int i) const {
       return fArray[off(i)];
     }

     inline /* constexpr */ T apply(unsigned int i) const {
       return fArray[off(i)];
     }

     inline T* Array() { return fArray; }  

     inline const T* Array() const { return fArray; }  

      /**
         assignment : only symmetric to symmetric allowed
       */
      template <class R>
      inline MatRepSym<T, D>& operator=(const R&) {
         STATIC_CHECK(0==1,
                      Cannot_assign_general_to_symmetric_matrix_representation);
         return *this;
      }
      inline MatRepSym<T, D>& operator=(const MatRepSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs.Array()[i];
         return *this;
      }

      /**
         self addition : only symmetric to symmetric allowed
       */
      template <class R>
      inline MatRepSym<T, D>& operator+=(const R&) {
         STATIC_CHECK(0==1,
                      Cannot_add_general_to_symmetric_matrix_representation);
         return *this;
      }
      inline MatRepSym<T, D>& operator+=(const MatRepSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] += rhs.Array()[i];
         return *this;
      }

      /**
         self subtraction : only symmetric to symmetric allowed
       */
      template <class R>
      inline MatRepSym<T, D>& operator-=(const R&) {
         STATIC_CHECK(0==1,
                      Cannot_substract_general_to_symmetric_matrix_representation);
         return *this;
      }
      inline MatRepSym<T, D>& operator-=(const MatRepSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] -= rhs.Array()[i];
         return *this;
      }
      template <class R> 
      inline bool operator==(const R& rhs) const {
         bool rc = true;
         for(unsigned int i=0; i<D*D; ++i) {
            rc = rc && (operator[](i) == rhs[i]);
         }
         return rc;
      }

      enum {
         /// return no. of matrix rows
         kRows = D,
         /// return no. of matrix columns
         kCols = D,
         /// return no of elements: rows*columns
         kSize = D*(D+1)/2
      };


#ifdef SMATRIX_USE_CONSTEXPR
     static constexpr int off0(int i) { return i==0 ? 0 : off0(i-1)+i;} 
     static constexpr int off2(int i, int j) { return j<i ? off0(i)+j : off0(j)+i; }
     static constexpr int off1(int i) { return off2(i/D, i%D);}

     static int off(int i) {
       static constexpr auto v = rowOffsetsUtils::make<D*D>(off1);
       return v[i];
     }

     static inline constexpr unsigned int
     offset(unsigned int i, unsigned int j)
     {
       //if (j > i) std::swap(i, j);
       return off(i*D+j);
       // return (i>j) ? (i * (i+1) / 2) + j :  (j * (j+1) / 2) + i;
     }
#elif defined(SMATRIX_USE_COMPUTATION)
     static inline unsigned int
     offset(unsigned int i, unsigned int j)
     {
       return (i>j) ? (i * (i+1) / 2) + j :  (j * (j+1) / 2) + i;
     }
     static inline unsigned int 
     off(unsigned int i) {
       return offset(i/D, i%D);
     }
#else
     unsigned int
     offset(unsigned int i, unsigned int j) const
     {
       return Offsets()(i,j);
     }
     unsigned int 
     off(unsigned int i) const {
       return Offsets().apply(i);
     }


     void CreateOffsets() {
       const static RowOffsets<D> off;
       fOff = &off;
     }

     inline const RowOffsets<D> & Offsets() const {
       return *fOff;
     }
#endif


   private:
      //T __attribute__ ((aligned (16))) fArray[kSize];
      T fArray[kSize];

#if !defined(SMATRIX_USE_CONSTEXPR) && !defined(SMATRIX_USE_COMPUTATION)
      const RowOffsets<D> * fOff;   //! transient
#endif
   };



} // namespace Math
} // namespace ROOT


#endif // MATH_MATRIXREPRESENTATIONSSTATIC_H
