// $Id$

const char *PC1CVSID = "$Id$";
/* ----------------------------------------------------------------------- *
 *                                                                         *
 * PC1.cc                                                                  *
 *                                                                         *
 * C++ adaptation of PC1 implementation written by Alexander PUKALL 1991.  *
 *                                                                         *
 * Reference:  http://membres.lycos.fr/pc1/                                *
 *                                                                         *
 * Description:                                                            *
 * PC1 Cipher Algorithm (Pukall Cipher 1) for encryption/decryption.       *
 * One-way hash for password encryption also provided.                     *
 *                                                                         *
 * Key length is 256 bits                                                  *
 *                                                                         *
 * Free code no restriction to use please include the name of the Author   *
 * in the final software                                                   *
 * Tested with Turbo C 2.0 for DOS and Microsoft Visual C++ 5.0 for Win 32 *
 *                                                                         *
 * Adapted by G. Ganis (g.ganis@cern.ch), January 2005                     *
 * ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "PC1.hh"

typedef unsigned short ushort;

// The following string is not a password nor a key, only a random data input
// used if the user password or key is too short
static unsigned char cleref[kPC1LENGTH] =
   {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
    'q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5'};

namespace PC1 {

/* ------------------------------------------------------------------- *
 *         Local static auxilliary functions                           *
 * ------------------------------------------------------------------- */
//____________________________________________________________________
static ushort code(ushort &ind, ushort &si, ushort &x1a2, ushort *x1a0)
{
   // Encoding
   ushort ax, bx, cx, dx;
   ushort tmp;

   dx = x1a2 + ind;
   ax = x1a0[ind];
   cx = 0x015a;
   bx = 0x4e35;

   tmp = ax;
   ax  = si;
   si  = tmp;

   tmp = ax;
   ax  = dx;
   dx  = tmp;

   if (ax != 0) {
      ax=ax*bx;
   }

   tmp = ax;
   ax  = cx;
   cx  = tmp;

   if (ax != 0) {
      ax = ax*si;
      cx = ax+cx;
   }

   tmp = ax;
   ax  = si;
   si  = tmp;
   ax  = ax*bx;
   dx  = cx+dx;

   ax += 1;

   x1a2 = dx;
   x1a0[ind] = ax;

   ushort res = ax^dx;
   ind += 1;
   return res;
}

//____________________________________________________________________
static void assemble(unsigned char *cle, ushort &inter,
                     ushort &si, ushort &x1a2)
{
   // Assembling
   ushort ind = 0;
   ushort x1a0[kPC1LENGTH/2];

   x1a0[0] = cle[0]*256 + cle[1];
   ushort res = code(ind,si,x1a2,x1a0);
   inter = res;

   int j;
   for (j = 0; j < 15; j++) {
      x1a0[j+1] = x1a0[j]^(cle[2*(j+1)]*256 + cle[2*(j+1)+1]);
      res = code(ind,si,x1a2,x1a0);
      inter = inter^res;
   }
}

} // namespace PC1

/* ------------------------------------------------------------------- *
 *                          Public functions                           *
 * ------------------------------------------------------------------- */

//______________________________________________________________________
int PC1HashFun(const char *in, int lin, const char *sa, int lsa,
                                                        int it, char *out)
{
   // One-way hash function.
   // Calculate hash of lin bytes at in (max kPC1LENGTH), using salt
   // sa (max length kPC1LENGTH bytes).
   // Number of iterations to finalize is 'it'.
   // If the salt is not given, the buffer in is used as salt.
   // The output buffer must be allocated by the caller to contain at
   // least 2*kPC1LENGTH+1 bytes.
   // Return length of hash in bytes or -1 in case of wrong / incomplete
   // input arguments

   // Check inputs
   if (!in || lin <= 0 || !out)
      return -1;

   //
   // Declarations and initializations
   unsigned char bin[kPC1LENGTH];
   unsigned char cle[kPC1LENGTH];
   unsigned char tab[kPC1LENGTH] = {0};
   unsigned int ix = 0;
   int j = 0;
   memset((void *)&bin[0],0,kPC1LENGTH);
   memset((void *)&tab[0],0,kPC1LENGTH);

   //
   // Fill bin with the first lin bytes of in
   int lbin = (lin > kPC1LENGTH) ? kPC1LENGTH : lin;
   memcpy(&bin[0],in,lbin);

   //
   // Fill cle ...
   int lcle = 0;
   if (sa && lsa > 0) {
      // ... with the salt
      for( j = 0; j < lsa; j++) {
         cle[j] = sa[j];
      }
      lcle = lsa;
   } else {
      // salt not given: use the password itself
      for( j = 0; j < lin; j++) {
         cle[j] = in[j];
      }
      lcle = lin;
   }
   //
   // If too short, complete cle with the ref string 
   for( j = lcle; j < kPC1LENGTH; j++) {
      cle[j] = cleref[j];
   }

   //
   // First round
   ushort si = 0;
   ushort inter = 0;
   ushort x1a2= 0;
   for (j = 0; j < kPC1LENGTH; j++) {
      short c = bin[j];
      PC1::assemble(cle,inter,si,x1a2);
      ushort cfc = inter >> 8;
      ushort cfd = inter & 0xFF; // cfc^cfd = random byte
      int k = 0;
      for( ; k < kPC1LENGTH; k++) {
         cle[k] = cle[k]^c;
      }
      c = c^(cfc^cfd);
      tab[ix] = tab[ix]^c;
      ix += 1;
      if (ix >= kPC1LENGTH) ix = 0;
   }

   //
   // Second round
   // to avoid dictionary attack on the passwords
   // int it = 63254;
   for (j = 1; j <= it; j++) {
      short c = tab[ix];
      PC1::assemble(cle,inter,si,x1a2);
      ushort cfc = inter >> 8;
      ushort cfd = inter & 0xFF; // cfc^cfd = random byte
      int k = 0;
      for (; k < kPC1LENGTH; k++) {
         cle[k] = cle[k]^c;
      }
      c = c^(cfc^cfd);
      tab[ix] = tab[ix]^c;
      ix += 1;
      if (ix >= kPC1LENGTH) ix = 0;
   }

   //
   // Prepare output 
   int k = 0;
   for( j = 0; j < kPC1LENGTH; j++) {
      // we split the 'c' crypted byte into two 4 bits
      // parts 'd' and 'e'
      short d = (tab[j] >> 4);
      short e = (tab[j] & 15);
      // write the two 4 bits parts in the output buffer
      // as text range from A to P
      out[k++] = d + 0x61;
      out[k++] = e + 0x61;
   }
   // Null terminated
   out[k] = 0;

   // Return buffer length
   return k;
}

//______________________________________________________________________
int PC1Encrypt(const char *in, int lin, const char *key, int lkey, char *out)
{
   // Encrypting routine.
   // Encode lin bytes at in using key (max key length kPC1LENGTH).
   // The output buffer must be allocated by the caller to contain at
   // least 2*lin bytes.
   // Return length of encrypted string in bytes or -1 in case of
   // wrong / incomplete input arguments

   // Check inputs
   if (!in || lin <= 0 || !key || lkey <= 0 || !out)
      return -1;

   //
   // Declarations and initializations
   unsigned char cle[kPC1LENGTH];
   int j = 0;

   //
   // Fill cle with the given key
   int lk = (lkey > kPC1LENGTH) ? kPC1LENGTH : lkey;
   for( j = 0; j < lk; j++) {
      cle[j] = key[j];
   }
   //
   // If too short, complete with the ref string 
   for( j = lk; j < kPC1LENGTH; j++) {
      cle[j] = cleref[j];
   }

   //
   // Define internal variables
   ushort si = 0;
   ushort inter = 0;
   ushort x1a2= 0;
   //
   // Encrypt
   int n = 0;
   for (j = 0; j < lin; j++) {

      short c = in[j];
      PC1::assemble(cle,inter,si,x1a2);
      ushort cfc = inter >> 8;
      ushort cfd = inter & 0xFF; // cfc^cfd = random byte
      int k = 0;
      for( ; k < kPC1LENGTH; k++) {
         cle[k] = cle[k]^c;
      }
      c = c^(cfc^cfd);

      // we split the 'c' crypted byte into two 4 bits
      // parts 'd' and 'e'
      short d = (c >> 4);
      short e = (c & 15);

      // write the two 4 bits parts in the output buffer
      // as text range from A to P
      out[n++] = d + 0x61;
      out[n++] = e + 0x61;
   }

   // Return buffer length
   return n;
}

//______________________________________________________________________
int PC1Decrypt(const char *in, int lin, const char *key, int lkey, char *out)
{
   // Decrypting routine.
   // Decrypt lin bytes at in using key (max key length kPC1LENGTH).
   // The output buffer must be allocated by the caller to contain at
   // least lin/2 bytes.
   // Return length of decrypted string in bytes or -1 in case of
   // wrong / incomplete input arguments

   // Check inputs
   if (!in || lin <= 0 || !key || lkey <= 0 || !out)
      return -1;

   //
   // Declarations and initializations
   unsigned char cle[kPC1LENGTH];
   int j = 0;

   //
   // Fill cle with the given key
   int lk = (lkey > kPC1LENGTH) ? kPC1LENGTH : lkey;
   for( j = 0; j < lk; j++) {
      cle[j] = key[j];
   }
   //
   // If too short, complete with the ref string 
   for( j = lk; j < kPC1LENGTH; j++) {
      cle[j] = cleref[j];
   }

   //
   // Define internal variables
   ushort si = 0;
   ushort inter = 0;
   ushort x1a2= 0;
   //
   // Decrypt
   int n = 0;
   for (j = 0; j < lin; j += 2 ) {

      short d = in[j]   - 0x61;
      short e = in[j+1] - 0x61;
      d = d << 4;
      short c = d + e;

      PC1::assemble(cle,inter,si,x1a2);
      ushort cfc = inter >> 8;
      ushort cfd = inter & 0xFF; // cfc^cfd = random byte

      c = c^(cfc^cfd);
      int k = 0;
      for( ; k < kPC1LENGTH; k++) {
         cle[k] = cle[k]^c;
      }
      out[n++] = c;
   }

   // Return buffer length
   return n;
}
