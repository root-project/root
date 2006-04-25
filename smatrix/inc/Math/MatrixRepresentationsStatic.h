// @(#)root/smatrix:$Name:  $:$Id: MatrixRepresentationsStatic.h,v 1.5 2006/03/17 15:11:35 moneta Exp $
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
    
    
//     template<unigned int D>
//     struct Creator { 
//       static const RowOffsets<D> & Offsets() {
// 	static RowOffsets<D> off;
// 	return off;
//       }



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
