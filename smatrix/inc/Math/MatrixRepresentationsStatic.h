// @(#)root/smatrix:$Name:  $:$Id: MatrixRepresentationsStatic.h,v 1.3 2006/03/08 15:11:06 moneta Exp $
// Authors: L. Moneta, J. Palacios    2006  

#ifndef ROOT_Math_MatrixRepresentationsStatic_h
#define ROOT_Math_MatrixRepresentationsStatic_h 1

// Include files

/** @class MatrixRepresentationsStatic MatrixRepresentationsStatic.h Math/MatrixRepresentationsStatic.h
 *  
 *
 *  @author Juan Palacios
 *  @date   2006-01-15
 *
 *  Classes MatRepStd and MatRepSym for gneeric and symmetric matrix
 *  data storage and manipulation. Define data storage and access, plus
 *  operators =, +=, -=, ==.
 *
 */
#include <iostream>
#include "Math/StaticCheck.h"

namespace ROOT {
  namespace Math {

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
        for(unsigned int i=0; i<D1*D1; ++i) {
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
      T fArray[kSize];
    };
    
    
    template<unsigned int D>
    struct RowOffsets {
      RowOffsets() {
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
      int operator()(unsigned int i, unsigned int j) const { return fOff[i*D+j]; }
      int apply(unsigned int i) const { return fOff[i]; }
      int fOff[D*D];
    };

    // offset specializations for small matrix sizes
    // offset specializations for small matrix sizes

    int ind2[4] = { 0 , 1  ,  1 , 2   };
    template<>
    struct RowOffsets<2> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind2[i*2+j]; }
      inline int apply(unsigned int i) const { return ind2[i]; }
    };

    int ind3[9] = { 0 , 1 , 3 , 1 , 2 , 4 , 3 , 4 , 5   };
    template<>
    struct RowOffsets<3> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind3[i*3+j]; }
      inline int apply(unsigned int i) const { return ind3[i]; }
    };

    int ind4[16] = { 0 , 1 , 3 , 6 ,  1 , 2 , 4 , 7  ,  3 , 4 , 5 , 8  ,  6 , 7 , 8 , 9   };
    template<>
    struct RowOffsets<4> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind4[i*4+j]; }
      inline int apply(unsigned int i) const { return ind4[i]; }
    };

    int ind5[25] = { 0 , 1 , 3 , 6 , 10 ,  1 , 2 , 4 , 7 , 11 ,  3 , 4 , 5 , 8 , 12 ,  6 , 7 , 8 , 9 , 13 ,  10 , 11 , 12 , 13 , 14   };                       
    template<>
    struct RowOffsets<5> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind5[i*5+j]; }
      inline int apply(unsigned int i) const { return ind5[i]; }
    };

    int ind6[36] = { 0 , 1 , 3 , 6 , 10 , 15 , 1 , 2 , 4 , 7 , 11 , 16 ,  3 , 4 , 5 , 8 , 12 , 17 ,  6 , 7 , 8 , 9 , 13 , 18 ,  10 , 11 , 12 , 13 , 14 , 19 ,  15 , 16 , 17 , 18 , 19 , 20   };
    template<>
    struct RowOffsets<6> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind6[i*6+j]; }
      inline int apply(unsigned int i) const { return ind6[i]; }
    };

    int ind7[49] = { 0 , 1 , 3 , 6 , 10 , 15 , 21 , 1 , 2 , 4 , 7 , 11 , 16 , 22 , 3 , 4 , 5 , 8 , 12 , 17 , 23 , 6 , 7 , 8 , 9 , 13 , 18 , 24 , 10 , 11 , 12 , 13 , 14 , 19 , 25 , 15 , 16 , 17 , 18 , 19 , 20 , 26 , 21 , 22 , 23 , 24 , 25 , 26 , 27 };
    template<>
    struct RowOffsets<7> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind7[i*7+j]; }
      inline int apply(unsigned int i) const { return ind7[i]; }
    };

    int ind8[64] = { 0 , 1 , 3 , 6 , 10 , 15 , 21 , 28 , 1 , 2 , 4 , 7 , 11 , 16 , 22 , 29 , 3 , 4 , 5 , 8 , 12 , 17 , 23 , 30 , 6 , 7 , 8 , 9 , 13 , 18 , 24 , 31 , 10 , 11 , 12 , 13 , 14 , 19 , 25 , 32 , 15 , 16 , 17 , 18 , 19 , 20 , 26 , 33 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 34 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 };
    template<>
    struct RowOffsets<8> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind8[i*8+j]; }
      inline int apply(unsigned int i) const { return ind8[i]; }
    };

    int ind9[81] = { 0 , 1 , 3 , 6 , 10 , 15 , 21 , 28 , 36 , 1 , 2 , 4 , 7 , 11 , 16 , 22 , 29 , 37 , 3 , 4 , 5 , 8 , 12 , 17 , 23 , 30 , 38 , 6 , 7 , 8 , 9 , 13 , 18 , 24 , 31 , 39 , 10 , 11 , 12 , 13 , 14 , 19 , 25 , 32 , 40 , 15 , 16 , 17 , 18 , 19 , 20 , 26 , 33 , 41 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 34 , 42 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 43 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 };
    template<>
    struct RowOffsets<9> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind9[i*9+j]; }
      inline int apply(unsigned int i) const { return ind9[i]; }
    };

    int ind10[100] = { 0 , 1 , 3 , 6 , 10 , 15 , 21 , 28 , 36 , 45 , 1 , 2 , 4 , 7 , 11 , 16 , 22 , 29 , 37 , 46 , 3 , 4 , 5 , 8 , 12 , 17 , 23 , 30 , 38 , 47 , 6 , 7 , 8 , 9 , 13 , 18 , 24 , 31 , 39 , 48 , 10 , 11 , 12 , 13 , 14 , 19 , 25 , 32 , 40 , 49 , 15 , 16 , 17 , 18 , 19 , 20 , 26 , 33 , 41 , 50 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 34 , 42 , 51 , 28 , 29 , 30 , 31 , 32 , 33 , 34 , 35 , 43 , 52 , 36 , 37 , 38 , 39 , 40 , 41 , 42 , 43 , 44 , 53 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 };
    template<>
    struct RowOffsets<10> {
      inline int operator()(unsigned int i, unsigned int j) const { return ind10[i*10+j]; }
      inline int apply(unsigned int i) const { return ind10[i]; }
    };


    template <class T, unsigned int D>
    class MatRepSym {

    public: 

      typedef T  value_type;

      inline const T& operator()(unsigned int i, unsigned int j) const {
        return fArray[Offsets()(i,j)];
      }
      inline T& operator()(unsigned int i, unsigned int j) {
        return fArray[Offsets()(i,j)];
      }

      inline T& operator[](unsigned int i) { 
	return fArray[Offsets().apply(i) ];
      }

      inline const T& operator[](unsigned int i) const {
	return fArray[Offsets().apply(i) ];
     }

      inline T apply(unsigned int i) const {
	return fArray[Offsets().apply(i) ];
        //return operator()(i/D, i%D);
      }

      inline T* Array() { return fArray; }  

      inline const T* Array() const { return fArray; }  

      inline MatRepSym<T, D>& operator+=(const MatRepSym& rhs) {
        for(unsigned int i=0; i<kSize; ++i) fArray[i] += rhs.Array()[i];
        return *this;
      }

      inline MatRepSym<T, D>& operator-=(const MatRepSym& rhs) {
        for(unsigned int i=0; i<kSize; ++i) fArray[i] -= rhs.Array()[i];
        return *this;
      }
      template <class R>
      inline MatRepSym<T, D>& operator=(const R& rhs) {
        STATIC_CHECK(0==1,
                     Cannot_assign_general_to_symmetric_matrix_representation);
        return *this;
      }
      inline MatRepSym<T, D>& operator=(const MatRepSym& rhs) {
        for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs.Array()[i];
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

      
      inline static const RowOffsets<D> & Offsets() {
	static RowOffsets<D> off;
	return off;
      }

    private:
      T fArray[kSize];

    };


 
  } // namespace Math
} // namespace ROOT


#endif // MATH_MATRIXREPRESENTATIONSSTATIC_H
