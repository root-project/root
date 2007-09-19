// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

#ifndef ROOT_Math_GenVector_GenVector_exception 
#define ROOT_Math_GenVector_GenVector_exception  1

// ======================================================================
// $Id $
//
// Define the exception type used throughout this package.
// ======================================================================


// ----------------------------------------------------------------------
// Prolog

#include <stdexcept>
#include <string>


namespace ROOT {
namespace Math {

class GenVector_exception;
void Throw(GenVector_exception & e);

// ----------------------------------------------------------------------
// GenVector_exception class definition

class GenVector_exception
  : public std::runtime_error
{
public:
  GenVector_exception( const std::string & s )
    : runtime_error(s)
  { }

// Compiler-generated copy ctor, copy assignment, dtor are fine
// Inherited what() from runtime_error is fine

  static bool EnableThrow()  { bool tmp = fgOn; fgOn = true;  return tmp; }
  static bool DisableThrow() { bool tmp = fgOn; fgOn = false; return tmp; }

private:
  friend void Throw(GenVector_exception &);
  static bool fgOn;
  
};  // GenVector_exception


// ----------------------------------------------------------------------
// Epilog

void Throw(GenVector_exception & e); 

}  // namespace Math
}  // namespace ROOT

#endif // GENVECTOR_EXCEPTION_H
