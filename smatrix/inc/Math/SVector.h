// @(#)root/smatrix:$Name:  $:$Id: SVector.h,v 1.8 2006/04/20 13:13:21 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005  

#ifndef ROOT_Math_SVector
#define ROOT_Math_SVector
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   16. Mar 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: A fixed size Vector class
//
// changes:
// 16 Mar 2001 (TG) creation
// 21 Mar 2001 (TG) SVector::value_type added
// 21 Mar 2001 (TG) added operators +=, -=, *=, /=
// 26 Mar 2001 (TG) added place_at()
// 03 Apr 2001 (TG) Array() added
// 06 Apr 2001 (TG) CTORS added
// 07 Apr 2001 (TG) CTORS added
// 22 Aug 2001 (TG) CTOR(T*,len) added
// 04 Sep 2001 (TG) moved inlined functions to .icc file
// 14 Jan 2002 (TG) added operator==(), operator!=(), operator>(), operator<()
//
// ********************************************************************

#include "Math/MConfig.h"

#include <iosfwd>

// expression engine
#include "Math/Expression.h"




namespace ROOT { 

  namespace Math { 

//     template <class T, unsigned int D, unsigned int D2> class MatRepStd;

//     template <class A, class T, unsigned int D, unsigned int D2 = 1, class R = MatRepStd<T,D,D2> > class Expr;


/** 
    SVector: a generic fixed size Vector class.
    The class is template on the scalar type and on the vector size D. 



    @ingroup SMatrix
    @memo SVector
    @author T. Glebe
*/
//==============================================================================
// SVector
//==============================================================================
template <class T, unsigned int D>
class SVector {
public:
  /** @name --- Typedefs --- */
  ///
  typedef T  value_type;

  /** STL iterator interface. */
  typedef T*  iterator;

  /** STL const_iterator interface. */
  typedef const T*  const_iterator;
  
  /** @name --- Constructors --- */


  /** @name --- Constructors --- */
  /**
     Default constructor: vector filled with zero values 
   */
  SVector();
  ///
  template <class A>
  SVector(const VecExpr<A,T,D>& rhs);
  ///
  SVector(const SVector<T,D>& rhs);
  /// $D1\le D$ required!
  template <unsigned int D1>
  SVector(const SVector<T,D1>& rhs);
  /// $D1\le D-1$ required!
  template <unsigned int D1>
  SVector(const T& a1, const SVector<T,D1>& rhs);

  // new constructs using STL iterator interface
  // skip - need to solve the ambiguities 
#ifdef LATER
  /**
   * Constructor with STL iterator interface. The data will be copied into the vector
   */
  template<class InputIterator>
  explicit SVector(InputIterator begin, InputIterator end); 

  /**
   * Constructor with STL iterator interface. The data will be copied into the vector
   */
  template<class InputIterator>
  explicit SVector(InputIterator begin, unsigned int size); 

#else 
  // if you use iterator this is not necessary

  /// fill from array, len must be equal to D!
  SVector( const T *  a, unsigned int len);

  /// fill from iterators 
  //(iterator is T* to skip ambiguities)
  SVector(const_iterator begin, const_iterator end);

#endif
  ///
  SVector(const T& rhs);
  ///
  SVector(const T& a1, const T& a2);
  ///
  SVector(const T& a1, const T& a2, const T& a3);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4,
	  const T& a5);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4,
	  const T& a5, const T& a6);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4,
	  const T& a5, const T& a6, const T& a7);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4,
	  const T& a5, const T& a6, const T& a7, const T& a8);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4,
	  const T& a5, const T& a6, const T& a7, const T& a8,
	  const T& a9);
  ///
  SVector(const T& a1, const T& a2, const T& a3, const T& a4,
	  const T& a5, const T& a6, const T& a7, const T& a8,
	  const T& a9, const T& a10);

  

  ///
  SVector<T,D>& operator=(const T& rhs);
  ///
  template <class A>
  SVector<T,D>& operator=(const VecExpr<A,T,D>& rhs);


  enum {
    /// return vector size
    kSize = D
  };

  /** @name --- Access functions --- */

  /// return dimension $D$
  inline static unsigned int Dim() { return D; }
  /// access the parse tree. Index starts from zero
  T apply(unsigned int i) const;
  /// return read-only pointer to internal array
  const T* Array() const;
  /// return pointer to internal array
  T* Array();

  /** @name --- STL-like interface --- */


  /** STL iterator interface. */
  iterator begin();

  /** STL iterator interface. */
  iterator end();

  /** STL const_iterator interface. */
  const_iterator begin() const;

  /** STL const_iterator interface. */
  const_iterator end() const;


  /** @name --- Operators --- */
  /// element wise comparison
  bool operator==(const T& rhs) const;
  /// element wise comparison
  bool operator!=(const T& rhs) const;
  /// element wise comparison
  bool operator==(const SVector<T,D>& rhs) const;
  /// element wise comparison
  bool operator!=(const SVector<T,D>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator==(const VecExpr<A,T,D>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator!=(const VecExpr<A,T,D>& rhs) const;
  
  /// element wise comparison
  bool operator>(const T& rhs) const;
  /// element wise comparison
  bool operator<(const T& rhs) const;
  /// element wise comparison
  bool operator>(const SVector<T,D>& rhs) const;
  /// element wise comparison
  bool operator<(const SVector<T,D>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator>(const VecExpr<A,T,D>& rhs) const;
  /// element wise comparison
  template <class A>
  bool operator<(const VecExpr<A,T,D>& rhs) const;

  /// read-only access of vector elements. Index starts from 0. 
  const T& operator[](unsigned int i) const;
  /// read-only access of vector elements. Index starts from 0. 
  const T& operator()(unsigned int i) const;
  /// read/write access of vector elements. Index starts from 0. 
  T& operator[](unsigned int i);
  /// read/write access of vector elements. Index starts from 0. 
  T& operator()(unsigned int i);

  ///
  SVector<T,D>& operator+=(const SVector<T,D>& rhs);
  ///
  SVector<T,D>& operator-=(const SVector<T,D>& rhs);
  ///
  SVector<T,D>& operator*=(const SVector<T,D>& rhs);
  ///
  SVector<T,D>& operator/=(const SVector<T,D>& rhs);


#ifndef __CINT__
  ///
  template <class A>
  SVector<T,D>& operator+=(const VecExpr<A,T,D>& rhs);
  ///
  template <class A>
  SVector<T,D>& operator-=(const VecExpr<A,T,D>& rhs);
  ///
  template <class A>
  SVector<T,D>& operator*=(const VecExpr<A,T,D>& rhs);
  ///
  template <class A>
  SVector<T,D>& operator/=(const VecExpr<A,T,D>& rhs);

#endif

  /** @name --- Expert functions --- */
  /// transform vector into a vector of lenght 1
  SVector<T,D>& Unit();
  /// place a sub-vector starting at <row>
  template <unsigned int D2>
  SVector<T,D>& Place_at(const SVector<T,D2>& rhs, unsigned int row);
  /// place a sub-vector starting at <row>
  template <class A, unsigned int D2>
  SVector<T,D>& Place_at(const VecExpr<A,T,D2>& rhs, unsigned int row);

  /**
     return a subvector of size N starting at the value row
     where N is the size of the returned vector (SubVector::kSize)
     Condition  row+N <= D
   */ 
  template <class SubVector >  
  SubVector Sub(unsigned int row) const;


  /// used by operator<<()
  std::ostream& Print(std::ostream& os) const;

private:
  T fArray[D];
}; // end of class SVector


//==============================================================================
// operator<<
//==============================================================================
template <class T, unsigned int D>
std::ostream& operator<<(std::ostream& os, const ROOT::Math::SVector<T,D>& rhs);



  }  // namespace Math

}  // namespace ROOT



#ifndef __CINT__

// include implementation file
#include "Math/SVector.icc"

// include operators and functions
#include "Math/UnaryOperators.h"
#include "Math/BinaryOperators.h"
#include "Math/Functions.h"

#endif // __CINT__


#endif  /* ROOT_Math_SVector  */
