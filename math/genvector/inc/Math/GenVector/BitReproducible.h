// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG / FNAL ROOT MathLib Team                  *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class LorentzVector
//
// Created by: fischler  at Mon Jun 25  2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_BitReproducible
#define ROOT_Math_GenVector_BitReproducible  1

#include <iostream>
#include <string>
#include <exception>

#include <iomanip>

namespace ROOT {
 namespace Math {
  namespace GenVector_detail {

class BitReproducibleException  : public std::exception
{
public:
  BitReproducibleException(const std::string & w) noexcept : fMsg(w) {}
  ~BitReproducibleException() noexcept override {}
  const char *what() const noexcept override { return fMsg.c_str(); }
  private:
  std::string fMsg;
};  // DoubConvException

class BitReproducible {
public:

  // dto2longs(d, i1, i2) returns (in i1 and i2) two unsigned ints
  // representation of its double input.  This is byte-ordering
  // independent, and depends for complete portability ONLY on adherance
  // to the IEEE 754 standard for 64-bit floating point representation.
  // The first unsigned int contains the high-order bits in IEEE; thus
  // 1.0 will always be 0x3FF00000, 00000000
  static void Dto2longs(double d, unsigned int & i1, unsigned int & i2);

  // longs2double (i1,i2) returns a double containing the value represented by
  // its input, which must be a 2 unsigned ints.
  // The input is taken to be the representation according to
  // the IEEE 754 standard for a 64-bit floating point number, whose value
  // is returned as a double.  The byte-ordering of the double result is,
  // of course, tailored to the proper byte-ordering for the system.
  static double Longs2double (unsigned int i1, unsigned int i2);

  // dtox(d) returns a 16-character string containing the (zero-filled) hex
  // representation of its double input.  This is byte-ordering
  // independent, and depends for complete portability ONLY on adherance
  // to the IEEE 754 standard for 64-bit floating point representation.
  static std::string D2x(double d);

  static void Output ( std::ostream & os, double d ) {
    unsigned int i1, i2;
    Dto2longs(d, i1, i2);
    os << " " << i1 << " " << i2;
  }

  static void Input ( std::istream & is, double & d ) {
    unsigned int i1, i2;
    is >> i1 >> i2;
    d = Longs2double(i1, i2);
  }

  static void Output ( std::ostream & os, float f ) {
    unsigned int i1, i2;
    Dto2longs( double(f), i1, i2 );
    os << " " << i1 << " " << i2;
  }

  static void Input ( std::istream & is, float & f ) {
    unsigned int i1, i2;
    is >> i1 >> i2;
    f = float( Longs2double(i1, i2) );
  }


private:
  union DB8 {
    unsigned char fB[8];
    double fD;
  };
  static void Fill_byte_order ();
  static bool fgByte_order_known;
  static int  fgByte_order[8];
    // Meaning of byte_order:  The first (high-order in IEEE 754) byte to
    // output (or the high-order byte of the first unsigned int)
    // is  of db.b[byte_order[0]].  Thus the index INTO byte_order
    // is a position in the IEEE representation of the double, and the value
    // of byte_order[k] is an offset in the memory representation of the
    // double.

};  // BitReproducible

}  // namespace _GenVector_detail
}  // namespace Math
}  // namespace ROOT

// A note about floats and long doubles:
//
// BitReproducible can be used with floats by doing the equivalent of
//   float x = x0; BitReproducible::dto2longs (x, i, j);
//   float y = BitReproducible::longs2double (i, j);
// The results are correct.
// The only inefficiency is that two integers are used where one would suffice.
//
// The same artifice will compile for long double.  However, any value of the
// long double which is not precisely representable as a double will not
// give exact results for the read-back.
//
// We intend in the near future to create a templated version of this class
// which cures both the above flaws.  (In the case of long double, this is
// contingent upon finding some IEEE standard for the bits in a 128-bit double.)


#endif // DOUBCONV_HH
