#include "ROOT/RConfig.hxx"
#include "ROOT/RCryptoRandom.hxx"

#ifdef WIN32
#include "Windows4Root.h"
#include <bcrypt.h>
#else
#if defined(R__ARC4_STDLIB)
#include <cstdlib>
#elif defined(R__ARC4_BSDLIB)
#include <bsd/stdlib.h>
#elif defined(R__GETRANDOM_CLIB)
#include <sys/random.h>
#elif defined(R__USE_URANDOM)
#include <fstream>
#endif
#endif

bool ROOT::Internal::GetCryptoRandom(void *buf, unsigned int len)
{
   if (len > 256)
      return false;

#ifdef WIN32

   return BCryptGenRandom((BCRYPT_ALG_HANDLE)NULL, (PUCHAR)buf, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;

#else // UNIX

#if defined(R__ARC4_STDLIB) || defined(R__ARC4_BSDLIB)
   arc4random_buf(buf, len);
   return true;
#elif defined(R__GETRANDOM_CLIB)
   return getrandom(buf, len, GRND_NONBLOCK) == len;
#elif defined(R__USE_URANDOM)
   std::ifstream urandom{"/dev/urandom"};
   if (!urandom)
      return false;
   urandom.read(reinterpret_cast<char *>(buf), len);
   return urandom.good();
#else
#error "Reliable cryptographic random function not defined"
   return false;
#endif

#endif
}
