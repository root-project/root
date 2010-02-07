// $Id$

const char *PC3CVSID = "$Id$";
/* ----------------------------------------------------------------------- *
 *                                                                         *
 * PC3.cc                                                                  *
 *                                                                         *
 * C++ adaptation of PKEP implementation written by Alexander PUKALL 1991. *
 *                                                                         *
 * PKEP ( Pukall Key Exchange Protocol (c) Alexander PUKALL 1997           *
 *                                                                         *
 * Reference:  http://membres.lycos.fr/pc1/                                *
 *                                                                         *
 * Description:                                                            *
 * Algorithm allowing the secure exchange of a random password using the   *
 * PC3 cipher for random number generation based on a 160-bit seed.        *
 * Initialization creates private and public parts; exponentiation builds  *
 * the key using the received public part.                                 *
 *                                                                         *
 * Created Key length is 256 bits (32 bytes). Input random string can be   *
 * up to 256 bytes, but 32 or 64 should be typically enough.               *
 * Buffers for private and public parts should be of length kPC3SLEN       *
 *                                                                         *
 * Fro the author:                                                         *
 * Free code no restriction to use please include the name of the Author   *
 * in the final software                                                   *
 *                                                                         *
 * Adapted by G. Ganis (g.ganis@cern.ch), February 2005                    *
 * ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "PC3.hh"

#define kMASKLAST (0x80000000)
#define kMASKFIRST (0x00000001)

namespace PC3 {

/* ------------------------------------------------------------------- *
 *         Local static auxilliary functions                           *
 * ------------------------------------------------------------------- */
//
//____________________________________________________________________
static unsigned int rotl(unsigned int n, unsigned int nl)
{
   // bit left rotation (VC++ _rotl)
   unsigned int i = 0;

   for ( i = 0; i < nl; i++) {
      bool bset = ((n & kMASKLAST) == kMASKLAST) ? 1 : 0;
      n <<= 1;
      if (bset)
         n |= kMASKFIRST;
      else
         n &= ~kMASKFIRST;
   }
   return n;
}

//____________________________________________________________________
static unsigned long stream(unsigned int &r1, unsigned long b1)
{
   static unsigned long a1 = 0x015a4e35;

   b1 = ( b1 * a1 ) + 1;
   r1 = rotl( (r1 + (( b1 >> 16 ) & 0x7fff)), (r1%16) );

   return b1;
}

//____________________________________________________________________
static uchar pc3stream(uchar byte, unsigned long *b1,
                       unsigned int &r1, unsigned int key)
{
   unsigned short d;

   unsigned long i = 0;
   for ( ; i<=(key-1); i++)
      b1[i] = stream(r1,b1[i]);

   d = byte;
   byte = byte^(r1 & 255);
   r1 += d;
   b1[key-1] = b1[key-1] + d;

   return byte;
}

//____________________________________________________________________
unsigned int pc3init(unsigned int lngkey, uchar *code,
                     unsigned long *b1, unsigned int &key)
{
   unsigned int z,y,x,i;
   uchar tab[kPC3MAXRPWLEN],plain;
   div_t reste;
   unsigned int r1 = 0;

   if (lngkey > kPC3MAXRPWLEN) lngkey = kPC3MAXRPWLEN;
   if (lngkey < 1) {
      lngkey = 1;
      strcpy((char *)code,"a");
   }

   x = lngkey;

   for ( i = 0; i < x; i++) {
      tab[i] = code[i];
   }

   reste = div(lngkey,2);
   key = reste.quot;
   if (reste.rem != 0) key += 1;

   y=0;
   for ( z = 0; z <= (key-1); z++) {
      if ( (z == (key-1)) && (reste.rem != 0) ) {
         b1[z]=code[y]*256;
      } else {
         b1[z] = (code[y]*256) + code[y+1];
         y = y+1;
      }
      y = y+1;
   }

   unsigned long ii = 0;
   for ( ; ii <= (key-1); ii++) {
      for( z = 0; z <= ii; z++)
         b1[ii] = stream(r1,b1[ii]);
   }

   for ( i = 0; i < x; i++) {
      plain = pc3stream(tab[i],b1,r1,key);
      tab[i] = tab[i]^plain;
   }
   i=i-1;
   for ( z = 1; z <= ((x+1)*10); z++) {
      plain = pc3stream(tab[i],b1,r1,key);
      tab[i] = tab[i]^plain;
      i++;
      if (i >= x) i = 0;
   }

   reste = div(lngkey,2);
   key   = reste.quot;
   if (reste.rem != 0) key += 1;

   for ( z = 0; z < 128; z++) {
      b1[z]=0;
   }

   y=0;
   for (z=0;z<=(key-1);z++)
   {
      if ( (z==(key-1))&&(reste.rem!=0) )
      {
         b1[z]=tab[y]*256;
      }
      else
      {
         b1[z]=(tab[y]*256)+tab[y+1];
         y=y+1;
      }
      y=y+1;
   }

   for (z=0;z<x;z++)
   {
      code[z]=0;
      tab[z]=0;
   }

   r1 = 0;
   for ( ii = 0; ii <= (key-1); ii++) {
      for ( z = 0; z <= ii; z++)
         b1[ii] = stream(r1,b1[ii]);
   }
   return r1;
}

//____________________________________________________________________
static void funA(uchar *x, uchar *y, int o)
{
   int d = 0;
   int v = kPC3SLEN;
   for( ; v--;) {
      d += x[v]+y[v]*o;
      x[v] = d;
      d = d>>8;
   }
}

//____________________________________________________________________
static void funS(uchar *x, uchar *m)
{
   int v = 0;
   for( ; (v < kPC3SLEN-1) && (x[v] == m[v]);)
      v++;
   if (x[v] >= m[v])
      funA(x,m,-1);
}

//____________________________________________________________________
static void funR(uchar *x)
{
   int d = 0;
   int v = 0;
   for ( ; v < kPC3SLEN; ) {
      d |= x[v];
      x[v++] = d/2;
      d = (d & 1) << 8;
   }
}

//____________________________________________________________________
static void funM(uchar *x, uchar *y, uchar *m)
{
   uchar X[1024],Y[1024];

   memcpy(X,x,kPC3SLEN);
   memcpy(Y,y,kPC3SLEN);
   memset(x,0,kPC3SLEN);

   int z = kPC3SLEN*8;
   for( ; z--; ) {
      if (X[kPC3SLEN-1] & 0x1) {
         funA(x,Y,1);
         funS(x,m);
      }
      funR(X);
      funA(Y,Y,1);
      funS(Y,m);
   }
}

//____________________________________________________________________
static int createkey(uchar *rpwd, unsigned int lrpw, uchar *priv)
{
   // Create key
   uchar inite[64]={0x94,0x05,0xF4,0x50,0x81,0x79,0x38,0xAB,
                    0x39,0x81,0x05,0x8C,0xCD,0xE8,0x04,0xDF,
                    0x6E,0x7C,0xAB,0x07,0x63,0xFE,0x4A,0xD7,
                    0x47,0x05,0x9D,0x2D,0x73,0xA9,0x38,0xBA,
                    0xB5,0x48,0x39,0x10,0x0A,0xD8,0xD1,0x5A,
                    0x9D,0x64,0x74,0xF8,0x8B,0xC5,0x3E,0x9A,
                    0xBF,0x27,0x55,0x9C,0x0C,0x6A,0x7E,0xD8,
                    0xA4,0x78,0x96,0x4C,0x96,0xBB,0x3A,0xC3};
   unsigned long b1[128] = {0};
   uchar code[kPC3MAXRPWLEN];

   // Check inputs
   if (!rpwd || (lrpw <= 0) || !priv) {
      return -1;
   }

   // Check length
   lrpw = (lrpw > (kPC3MAXRPWLEN-2)) ? (kPC3MAXRPWLEN-2) : lrpw;
   unsigned int i = 0;
   for ( ; i < lrpw; i++)
      code[i] = rpwd[i];
   //
   // The last two chars must be 0
   code[lrpw] = '\0';
   code[lrpw+1] = '\0';

   unsigned int key = 0;
   unsigned int r1 = pc3init(lrpw+2,code,b1,key);

   for ( i = 1; i < kPC3SLEN; i++)
      priv[i-1] = pc3stream(inite[i-1],b1,r1,key);

   // We are done
   return 0;
}

} // namespace PC3

/* ------------------------------------------------------------------- *
 *                          Public functions                           *
 * ------------------------------------------------------------------- */
//____________________________________________________________________
int PC3InitDiPuk(uchar *rpwd, unsigned int lrpw, uchar *pub, uchar *priv)
{
   // Initialize public-private key computation (Phase 1).
   // Input:
   //        rpwd    buffer containing random bits
   //                (max size 256 bytes; advised >= 32)
   //        lrpw    number of meaningful bytes in rpwd
   //
   // Output:
   //        priv    buffer of size kPC3SLEN containing the
   //                information to keep locally secure
   //        pub     buffer of size kPC3SLEN containing the
   //                information to send to the counterpart
   //
   // Return 0 if OK, -1 if buffers undefined
   //
   // nb: priv and pub buffers must be allocated by the caller.
   //

   // Check inputs
   if (!rpwd || (lrpw <= 0) || !pub || !priv) {
      return -1;
   }

   uchar prime512[64] = {
      0xF5, 0x2A, 0xFF, 0x3C, 0xE1, 0xB1, 0x29, 0x40,
      0x18, 0x11, 0x8D, 0x7C, 0x84, 0xA7, 0x0A, 0x72,
      0xD6, 0x86, 0xC4, 0x03, 0x19, 0xC8, 0x07, 0x29,
      0x7A, 0xCA, 0x95, 0x0C, 0xD9, 0x96, 0x9F, 0xAB,
      0xD0, 0x0A, 0x50, 0x9B, 0x02, 0x46, 0xD3, 0x08,
      0x3D, 0x66, 0xA4, 0x5D, 0x41, 0x9F, 0x9C, 0x7C,
      0xBD, 0x89, 0x4B, 0x22, 0x19, 0x26, 0xBA, 0xAB,
      0xA2, 0x5E, 0xC3, 0x55, 0xE9, 0x2A, 0x05, 0x5F
   };
   uchar e[kPC3SLEN+1] = {0};
   uchar m[kPC3SLEN+1] = {0};
   uchar g[kPC3SLEN+1] = {0};

   g[kPC3SLEN-1] = 3;
   unsigned int pr = 1;
   for ( ; pr < kPC3SLEN; pr++)
      m[pr] = prime512[pr-1];

   if (PC3::createkey(rpwd,lrpw,priv) < 0)
      return -1;

   for ( pr = 1; pr < kPC3SLEN; pr++)
      e[pr] = priv[pr-1];

   uchar b[kPC3SLEN] = {0};
   b[kPC3SLEN-1] = 1;
   int n = kPC3SLEN*8;
   for( ; n--; ) {
      if ((e[kPC3SLEN-1] & 0x1))
         PC3::funM(b,g,m);
      PC3::funM(g,g,m);
      PC3::funR(e);
   }

   // Fill the public part
   int i = 1;
   for ( ; i < kPC3SLEN; i++)
      pub[i-1] = b[i];
   pub[kPC3SLEN-1] = 0;

   // We are done
   return 0;
}

//____________________________________________________________________
int PC3DiPukExp(uchar *pub, uchar *priv, uchar *key)
{
   // Compute key using buffer received from counterpart(Phase 2).
   // Input:
   //        priv    buffer of size kPC3SLEN generated by a
   //                previous call to PC3InitDiPuk.
   //        pub     buffer of size kPC3SLEN containing the
   //                public information of the counterpart
   //
   // Output:
   //        key     buffer of size kPC3KEYLEN containing the
   //                computed random key
   //
   // Return 0 if OK, -1 if buffers undefined
   //
   // nb: all buffers must be allocated by the caller.
   //

   // Check inputs
   if (!key || !pub || !priv) {
      return -1;
   }

   uchar prime512[64] = {
      0xF5, 0x2A, 0xFF, 0x3C, 0xE1, 0xB1, 0x29, 0x40,
      0x18, 0x11, 0x8D, 0x7C, 0x84, 0xA7, 0x0A, 0x72,
      0xD6, 0x86, 0xC4, 0x03, 0x19, 0xC8, 0x07, 0x29,
      0x7A, 0xCA, 0x95, 0x0C, 0xD9, 0x96, 0x9F, 0xAB,
      0xD0, 0x0A, 0x50, 0x9B, 0x02, 0x46, 0xD3, 0x08,
      0x3D, 0x66, 0xA4, 0x5D, 0x41, 0x9F, 0x9C, 0x7C,
      0xBD, 0x89, 0x4B, 0x22, 0x19, 0x26, 0xBA, 0xAB,
      0xA2, 0x5E, 0xC3, 0x55, 0xE9, 0x2A, 0x05, 0x5F
   };
   uchar e[kPC3SLEN+1] = {0};
   uchar m[kPC3SLEN+1] = {0};
   uchar g[kPC3SLEN+1] = {0};
   uchar b[kPC3SLEN+1] = {0};

   unsigned int pr = 1;
   for ( ; pr < kPC3SLEN; pr++) {
      g[pr] = pub[pr-1];
      e[pr] = priv[pr-1];
      m[pr] = prime512[pr-1];
   }
   b[kPC3SLEN-1] = 1;
   int n = kPC3SLEN*8;
   for ( ; n--;) {
      if ((e[kPC3SLEN-1] & 0x1))
         PC3::funM(b,g,m);
      PC3::funM(g,g,m);
      PC3::funR(e);
   }

   // Fill the key
   int i = 0;
   for ( ; i < kPC3KEYLEN; i++)
      key[i] = 0;
   for ( i = 1; i < kPC3SLEN; i++) {
      key[i%kPC3KEYLEN] = key[i%kPC3KEYLEN]^b[i];
   }

   // We are done
   return 0;
}
