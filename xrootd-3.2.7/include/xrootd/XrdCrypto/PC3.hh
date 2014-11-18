// $Id$
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

typedef unsigned char uchar;

#define kPC3SLEN      33
#define kPC3MAXRPWLEN 256
#define kPC3MINBITS   128
#define kPC3KEYLEN    32

int PC3InitDiPuk(uchar *rpwd, unsigned int lrpw, uchar *pub, uchar *priv);
int PC3DiPukExp(uchar *pub, uchar *priv, uchar *key);
