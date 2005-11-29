// @(#)root/smatrix:$Name:  $:$Id: Dinv.h,v 1.1 2005/11/24 16:03:42 brun Exp $
// Authors: T. Glebe, L. Moneta    2005  

#ifndef  ROOT_Math_Dinv
#define  ROOT_Math_Dinv
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   03. Apr 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: square Matrix inversion
//              Code was taken from CERNLIB::kernlib dfinv function, translated
//              from FORTRAN to C++ and optimized.
//              n:    Order of the square matrix
//              idim: First dimension of array A
//
// changes:
// 03 Apr 2001 (TG) creation
//
// ********************************************************************
#include "Math/Dfactir.h"
#include "Math/Dfinv.h"


 
namespace ROOT { 

  namespace Math { 



/** Inverter.
    Class to specialize calls to Dinv. Dinv computes the inverse of a square
    matrix if dimension $idim$ and order $n$. The content of the matrix will be
    replaced by its inverse. In case the inversion fails, the matrix content is
    destroyed. Invert specializes Dinv by the matrix order. E.g. if the order
    of the matrix is two, the routine Inverter<2> is called which implements
    Cramers rule.

    @author T. Glebe
*/
//==============================================================================
// Inverter class
//==============================================================================
template <unsigned int idim, unsigned int n = idim>
class Inverter {
public:
  ///
  template <class Matrix>
  static bool Dinv(Matrix& rhs) {

#ifdef XXX
      if (n < 1 || n > idim) {
	return false;
      }
#endif

      /* Initialized data */
      static unsigned int work[n];
      for(unsigned int i=0; i<n; ++i) work[i] = 0;

      static typename Matrix::value_type det = 0;

      /* Function Body */
      
      /*  N.GT.3 CASES.  FACTORIZE MATRIX AND INVERT. */
      if (Dfactir<Matrix,n,idim>(rhs,det,work) == false) {
	std::cerr << "Dfactir failed!!" << std::endl;
	return false;
      }
      return Dfinv<Matrix,n,idim>(rhs,work);
  } // Dinv
}; // class Inverter


/** Inverter<0>.
    In case of zero order, do nothing.

    @author T. Glebe
*/
//==============================================================================
// Inverter<0>
//==============================================================================
template <>
class Inverter<0> {
public:
  ///
  template <class Matrix>
  inline static bool Dinv(Matrix& rhs) { return true; }
};
    

/** Inverter<1>.
    $1\times1$ (sub-)matrix. $a_{11} \to 1/a_{11}$

    @author T. Glebe
*/
//==============================================================================
// Inverter<1>
//==============================================================================
template <>
class Inverter<1> {
public:
  ///
  template <class Matrix>
  static bool Dinv(Matrix& rhs) {
    typename Matrix::value_type* a = rhs.Array();
    
    if (a[0] == 0.) {
      return false;
    }
    a[0] = 1. / a[0];
    return true;
  }
};


/** Inverter<2>.
    $2\times2$ (sub-)matrix. Use Cramers rule.

    @author T. Glebe
*/
//==============================================================================
// Inverter<2>: Cramers rule
//==============================================================================
template <>
class Inverter<2> {
public:
  ///
  template <class Matrix>
  static bool Dinv(Matrix& rhs) {

    typename Matrix::value_type* a = rhs.Array();
    typename Matrix::value_type det = a[0] * a[3] - a[2] * a[1];
    
    if (det == 0.) { return false; }

    typename Matrix::value_type s = 1. / det;
    typename Matrix::value_type c11 = s * a[3];

    a[2] = -s * a[2];
    a[1] = -s * a[1];
    a[3] =  s * a[0];
    a[0] = c11;
    return true;
  }
};


/** Inverter<3>.
    $3\times3$ (sub-)matrix. Use pivotisation.

    @author T. Glebe
*/
//==============================================================================
// Inverter<3>
//==============================================================================
template <>
class Inverter<3> {
public:
  ///
  template <class Matrix>
  static bool Dinv(Matrix& rhs) {

    typename Matrix::value_type* a = rhs.Array();

    static typename Matrix::value_type t1, t2, t3, temp, s;
    static typename Matrix::value_type c11, c12, c13, c21, c22, c23, c31, c32, c33, det;

  
    /*     COMPUTE COFACTORS. */
    c11 = a[4] * a[8] - a[7] * a[5];
    c12 = a[7] * a[2] - a[1] * a[8];
    c13 = a[1] * a[5] - a[4] * a[2];
    c21 = a[5] * a[6] - a[8] * a[3];
    c22 = a[8] * a[0] - a[2] * a[6];
    c23 = a[2] * a[3] - a[5] * a[0];
    c31 = a[3] * a[7] - a[6] * a[4];
    c32 = a[6] * a[1] - a[0] * a[7];
    c33 = a[0] * a[4] - a[3] * a[1];

    t1 = std::fabs(a[0]);
    t2 = std::fabs(a[1]);
    t3 = std::fabs(a[2]);
    
    /*     (SET TEMP=PIVOT AND DET=PIVOT*DET.) */
    if(t1 < t2) {
      if (t3 < t2) {
	/*        (PIVOT IS A21) */
	temp = a[1];
	det = c13 * c32 - c12 * c33;
      } else {
	/*     (PIVOT IS A31) */
	temp = a[2];
	det = c23 * c12 - c22 * c13;
      }
    } else {
      if(t3 < t1) {
	/*     (PIVOT IS A11) */
	temp = a[0];
	det = c22 * c33 - c23 * c32;
      } else {
	/*     (PIVOT IS A31) */
	temp = a[2];
	det = c23 * c12 - c22 * c13;
      }
    }

    if (det == 0.) {
      return false;
    }

    /*     SET ELEMENTS OF INVERSE IN A. */
    s = temp / det;
    a[0] = s * c11;
    a[3] = s * c21;
    a[6] = s * c31;
    a[1] = s * c12;
    a[4] = s * c22;
    a[7] = s * c32;
    a[2] = s * c13;
    a[5] = s * c23;
    a[8] = s * c33;
    return true;
  }
};


  }  // namespace Math

}  // namespace ROOT
          


#endif  /* ROOT_Math_Dinv */
