// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

#ifndef ROOT_Math_GenVector_GenVector_exception
#define ROOT_Math_GenVector_GenVector_exception 1

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
inline void Throw(GenVector_exception &e);
namespace GenVector {
inline void Throw(const char *);
}

// ----------------------------------------------------------------------
// GenVector_exception class definition
//
// This class needs to be entirely contained in this header, otherwise
// the interactive usage of entities such as ROOT::Math::PtEtaPhiMVector
// is not possible because of missing symbols.
// This is due to the fact that the Throw function is used in the inline
// code bu this function is implemented in the Genvector library.
class GenVector_exception : public std::runtime_error {
public:
   GenVector_exception(const std::string &s) : runtime_error(s) {}

   // Compiler-generated copy ctor, copy assignment, dtor are fine
   // Inherited what() from runtime_error is fine

   static bool EnableThrow()
   {
      bool tmp = GenVector_exception::IsOn();
      IsOn() = true;
      return tmp;
   }
   static bool DisableThrow()
   {
      bool tmp = GenVector_exception::IsOn();
      IsOn() = false;
      return tmp;
   }

private:
   friend void Throw(GenVector_exception &);
   friend void GenVector::Throw(const char *);

   static bool &IsOn()
   {
      static bool isOn = false;
      return isOn;
   };

}; // GenVector_exception

// ----------------------------------------------------------------------
// Epilog

/// throw explicity GenVector exceptions
inline void Throw(GenVector_exception &e)
{
   if (GenVector_exception::IsOn())
      throw e;
}

namespace GenVector {
/// function throwing exception, by creating internally a GenVector_exception only when needed
inline void Throw(const char *s)
{
   if (!GenVector_exception::IsOn())
      return;
   GenVector_exception e(s);
   throw e;
}
} // namespace GenVector

} // namespace Math
} // namespace ROOT

#endif // GENVECTOR_EXCEPTION_H
