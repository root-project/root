// $Id$
/* ----------------------------------------------------------------------- *
 *                                                                         *
 * PC1.hh                                                                  *
 *                                                                         *
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

// Basic length (of key, output hash, ...) in bytes
#define kPC1LENGTH 32

//
// Encode / Decode functions
int PC1Encrypt(const char *, int, const char *, int, char *);
int PC1Decrypt(const char *, int, const char *, int, char *);

//
// One-way hash
int PC1HashFun(const char *, int, const char *, int, int, char *);
