/// \file ROOT/RCryptoRandom.hxx
/// \ingroup Base
/// \date 2026-04-24

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCryptoRandom
#define ROOT_RCryptoRandom

namespace ROOT {
namespace Internal {

/// Get random bytes from the operating system's cryptographic random number generator
/// The requested number of bytes must not exceed 256.
bool GetCryptoRandom(void *buf, unsigned int len);

} // namespace Internal
} // namespace ROOT

#endif
