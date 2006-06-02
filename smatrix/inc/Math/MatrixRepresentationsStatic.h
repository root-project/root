// @(#)root/smatrix:$Name:  $:$Id: MatrixRepresentationsStatic.h,v 1.6 2006/04/25 13:54:01 moneta Exp $
// Authors: L. Moneta, J. Palacios    2006  

#ifndef ROOT_Math_MatrixRepresentationsStatic_h
#define ROOT_Math_MatrixRepresentationsStatic_h 1

// Include files

/** 
    @defgroup MatRep Matrix Storage Representation 
 
    @author Juan Palacios
    @date   2006-01-15
 
    Classes MatRepStd and MatRepSym for generic and symmetric matrix
    data storage and manipulation. Define data storage and access, plus
    operators =, +=, -=, ==.
 
 */
#include <iostream>
#include "Math/StaticCheck.h"

namespace ROOT {
  namespace Math {
    
    /**
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


    /**
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

      MatRepSym() :fOff(0) { CreateOffsets(); } 

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

      
      void CreateOffsets() {
	static RowOffsets<D> off;
	fOff = &off;
      }
      
      inline const RowOffsets<D> & Offsets() const {
	return *fOff;
      }

    private:
      T fArray[kSize];

      RowOffsets<D> * fOff;   //! transient

    };


 
  } // namespace Math
} // namespace ROOT


#endif // MATH_MATRIXREPRESENTATIONSSTATIC_H
