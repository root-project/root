// $Id: ValidityKey.h,v 1.11 2009-12-17 18:50:42 avalassi Exp $
#ifndef COOLKERNEL_VALIDITYKEY_H
#define COOLKERNEL_VALIDITYKEY_H

// Include files
#include "CoolKernel/types.h"

namespace cool {

  /** @class ValidityKey ValidityKey.h
   *
   *  Type definition ValidityKey for the unit of an interval of validity.
   *
   *  The initial implementation, consistent with that of the original
   *  common API, is a typedef to a 64-bit unsigned integer.
   *
   *  Eventually this may evolve into a class implementing a more complex
   *  interface (for instance, implementing a strict order relationship or,
   *  more simply, the asUInt64 and asAttributeList methods).
   *
   *  Because of problems with the Oracle and MySQL backend implementations,
   *  the 64th bit is not used and the range of allowed ValidityKey values
   *  is that of a 63-bit unsigned integer, i.e. [0, +9223372036854775807].
   *  This is enough to store time stamps with nanosecond precision over a
   *  ~300 year range; if interpreted as a number of nanoseconds since 1970
   *  Jan 1st, a ValidityKey extends beyond the year 2250, well after LHC.
   *
   *  For reference: the problem with Oracle is an OCI-22053 overflow error
   *  when reading back the value -9223372036854775808, making it impossible
   *  to use the full signed int64 range; the problem with MySQL is that
   *  unsigned int64 are internally interpreted as signed int64 for all
   *  mathematical operations (e.g. "+4 > +9223372036854775808" is true),
   *  making it impossible to use the full unsigned int64 range either.
   *  Later on, it was found that SQLite also has problems in [2^63, 2^64-1],
   *  as values are internally stored as signed int64 and queries may break.
   *
   *  @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   *  @date   2004-10-27
   */

  // ValidityKey range: unsigned int64 in [0, +9223372036854775807]
  // NOTE: If we ever turn this into a class, we MUST provide operator++ and
  // operator-- to allow 'stepping' through intervals without depending on the
  // actual granularity.
  typedef UInt63 ValidityKey;
  const ValidityKey ValidityKeyMin = UInt63Min; // 0
  const ValidityKey ValidityKeyMax = UInt63Max; // +9223372036854775807

}

#endif
