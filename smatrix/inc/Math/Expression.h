// @(#)root/smatrix:$Name:  $:$Id: Expression.hv 1.0 2005/11/24 12:00:00 moneta Exp $
// Authors: T. Glebe, L. Moneta    2005  

#ifndef ROOT_Math_Expression
#define ROOT_Math_Expression
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   19. Mar 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: Expression Template Elements for SVector
//
// changes:
// 19 Mar 2001 (TG) creation
// 20 Mar 2001 (TG) added rows(), cols() to Expr
// 21 Mar 2001 (TG) added Expr::value_type
// 11 Apr 2001 (TG) rows(), cols() replaced by rows, cols
// 10 Okt 2001 (TG) added print() and operator<<() for Expr class
//
// ********************************************************************

/** Expr.
    An Expression wrapper class.

    @memo Expr.
    @author T. Glebe
*/
//==============================================================================
// Expr: class representing SVector expressions
//=============================================================================

// originally references where used as data members of operation wrappers
// problem appear on Windows because original ref goes out of scope use then by value by default (like in tvmet)
//#define SMATRIX_USE_REFERENCES   


#include <iomanip>


namespace ROOT { 

  namespace Math { 



template <class ExprType, class T, unsigned int D, unsigned int D2 = 0>
class Expr {
public:
  typedef T  value_type;

  ///
  Expr(const ExprType& rhs) :
    rhs_(rhs) {}

  ///
  ~Expr() {}

  ///
  inline T apply(unsigned int i) const {
    return rhs_.apply(i);
  }

#ifdef OLD_IMPL
  ///
  static const unsigned int rows = D;
  ///
  static const unsigned int cols = D2;
#else
  // use enumerations
  enum { 
    ///
    rows = D, 
  ///
    cols = D2,
  };
#endif

  /// used by operator<<()
  std::ostream& print(std::ostream& os) const {
    os.setf(std::ios::right,std::ios::adjustfield);

    if(D2 == 0) {
      unsigned int i=0;
      for(; i<D-1; ++i) {
	os << apply(i) << ", ";
      }
      os << apply(i);
    } else {
      os << "[ ";
      for (unsigned int i=0; i < D; ++i) {
	for (unsigned int j=0; j < D2; ++j) {
	  os << std::setw(12) << apply(i*D2+j);
	  if ((!((j+1)%12)) && (j < D2-1))
	  os << std::endl << "         ...";
	}
	if (i != D - 1)
	os << std::endl  << "  ";
      }
      os << " ]";
    } // if D2==0

    return os;
  }

private:
  ExprType rhs_; // cannot be a reference!
};

//==============================================================================
// operator<<
//==============================================================================
template <class A, class T, unsigned int D1, unsigned int D2>
inline std::ostream& operator<<(std::ostream& os, const Expr<A,T,D1,D2>& rhs) {
  return rhs.print(os);
}

/** BinaryOp.
    A class representing binary operators in the parse tree.

    @memo BinaryOp
    @author T. Glebe
*/
//==============================================================================
// BinaryOp
//==============================================================================
template <class Operator, class LHS, class RHS, class T>
class BinaryOp {
public:
  ///
  BinaryOp( Operator /* op */, const LHS& lhs, const RHS& rhs) :
    lhs_(lhs), rhs_(rhs) {}

  ///
  ~BinaryOp() {}

  ///
  inline T apply(unsigned int i) const {
    return Operator::apply(lhs_.apply(i), rhs_.apply(i));
  }

protected:

#ifdef SMATRIX_USE_REFERENCES
  const LHS& lhs_;
  const RHS& rhs_;
#else
  const LHS lhs_;
  const RHS rhs_;
#endif

};


/** UnaryOp.
    A class representing unary operators in the parse tree.

    @memo UnaryOp
    @author T. Glebe
*/
//==============================================================================
// UnaryOp
//==============================================================================
template <class Operator, class RHS, class T>
class UnaryOp {
public:
  ///
  UnaryOp( Operator /* op */ , const RHS& rhs) :
    rhs_(rhs) {}

  ///
  ~UnaryOp() {}
  
  ///
  inline T apply(unsigned int i) const {
    return Operator::apply(rhs_.apply(i));
  }

protected:

#ifdef SMATRIX_USE_REFERENCES
  const RHS& rhs_;
#else
  const RHS rhs_;
#endif

};


/** Constant.
    A class representing constant expressions (literals) in the parse tree.

    @memo Constant
    @author T. Glebe
*/
//==============================================================================
// Constant
//==============================================================================
template <class T>
class Constant {
public:
  ///
  Constant( const T& rhs ) :
    rhs_(rhs) {}

  ///
  ~Constant() {}

  ///
  inline T apply(unsigned int /*i */ ) const { return rhs_; }

protected:

#ifdef SMATRIX_USE_REFERENCES
  const T& rhs_;
#else
  const T rhs_;
#endif

};



  }  // namespace Math

}  // namespace ROOT
          


#endif  /* ROOT_Math_Expression */
