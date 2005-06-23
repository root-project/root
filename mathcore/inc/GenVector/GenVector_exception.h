// @(#)root/mathcore:$Name:  $:$Id: GenVector_exception.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

#ifndef GENVECTOR_EXCEPTION_HH
#define GENVECTOR_EXCEPTION_HH

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

};  // GenVector_exception


// ----------------------------------------------------------------------
// Epilog


}  // namespace Math
}  // namespace ROOT

#endif // GENVECTOR_EXCEPTION_HH
