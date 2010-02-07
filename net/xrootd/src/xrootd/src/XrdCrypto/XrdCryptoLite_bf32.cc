/******************************************************************************/
/*                                                                            */
/*                 X r d C r y p t o L i t e _ b f 3 2 . c c                  */
/*                                                                            */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdCryptoLite_bf32CVSID = "$Id$";

#include "XrdCrypto/XrdCryptoLite.hh"

#ifdef R__SSL

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>

#include <openssl/blowfish.h>

#include "XrdOuc/XrdOucCRC.hh"
#include "XrdSys/XrdSysHeaders.hh"

/******************************************************************************/
/*              C l a s s   X r d C r y p t o L i t e _ b f 3 2               */
/******************************************************************************/
  
class XrdCryptoLite_bf32 : public XrdCryptoLite
{
public:

virtual int  Decrypt(const char *key,      // Decryption key
                     int         keyLen,   // Decryption key byte length
                     const char *src,      // Buffer to be decrypted
                     int         srcLen,   // Bytes length of src  buffer
                     char       *dst,      // Buffer to hold decrypted result
                     int         dstLen);  // Bytes length of dst  buffer

virtual int  Encrypt(const char *key,      // Encryption key
                     int         keyLen,   // Encryption key byte length
                     const char *src,      // Buffer to be encrypted
                     int         srcLen,   // Bytes length of src  buffer
                     char       *dst,      // Buffer to hold encrypted result
                     int         dstLen);  // Bytes length of dst  buffer

         XrdCryptoLite_bf32(const char deType) : XrdCryptoLite(deType, 4) {}
        ~XrdCryptoLite_bf32() {}
};

/******************************************************************************/
/*                               D e c r y p t                                */
/******************************************************************************/

int XrdCryptoLite_bf32::Decrypt(const char *key,
                                int         keyLen,
                                const char *src,
                                int         srcLen,
                                char       *dst,
                                int         dstLen)
{
   BF_KEY decKey;
   unsigned char ivec[8] = {0,0,0,0,0,0,0,0};
   unsigned int crc32;
   int ivnum = 0, dLen = srcLen-sizeof(crc32);

// Make sure we have data
//
   if (dstLen <= (int)sizeof(crc32) || dstLen < srcLen) return -EINVAL;

// Set the key
//
   BF_set_key(&decKey, keyLen, (const unsigned char *)key);

// Decrypt
//
   BF_cfb64_encrypt((const unsigned char *)src, (unsigned char *)dst, srcLen,
                    &decKey, ivec, &ivnum, BF_DECRYPT);

// Perform the CRC check to verify we have valid data here
//
   memcpy(&crc32, dst+dLen, sizeof(crc32));
   crc32 = ntohl(crc32);
   if (crc32 != XrdOucCRC::CRC32((const unsigned char *)dst, dLen))
      return -EPROTO;

// Return success
//
   return dLen;
}
  
/******************************************************************************/
/*                               E n c r y p t                                */
/******************************************************************************/

int XrdCryptoLite_bf32::Encrypt(const char *key,
                                int         keyLen,
                                const char *src,
                                int         srcLen,
                                char       *dst,
                                int         dstLen)
{
   BF_KEY encKey;
   unsigned char buff[4096], *bP, *mP = 0, ivec[8] = {0,0,0,0,0,0,0,0};
   unsigned int crc32;
   int ivnum = 0, dLen = srcLen+sizeof(crc32);

// Make sure that the destination if at least 4 bytes larger and we have data
//
   if (dstLen-srcLen < (int)sizeof(crc32) || srcLen <= 0) return -EINVAL;

// Normally, the msg is 4k or less but if more, get a new buffer
//
   if (dLen <= (int)sizeof(buff)) bP = buff;
      else {if (!(mP = (unsigned char *)malloc(dLen))) return -ENOMEM;
               else bP = mP;
           }

// Append a crc
//
   memcpy(bP, src, srcLen);
   crc32 = XrdOucCRC::CRC32(bP, srcLen);
   crc32 = htonl(crc32);
   memcpy((bP+srcLen), &crc32, sizeof(crc32));

// Set the key
//
   BF_set_key(&encKey, keyLen, (const unsigned char *)key);

// Encrypt
//
   BF_cfb64_encrypt(bP, (unsigned char *)dst, dLen,
                    &encKey, ivec, &ivnum, BF_ENCRYPT);

// Free temp buffer and return success
//
   if (mP) free(mP);
   return dLen;
}
#endif

/******************************************************************************/
/*                X r d C r y p t o L i t e _ N e w _ b f 3 2                 */
/******************************************************************************/
  
XrdCryptoLite *XrdCryptoLite_New_bf32(const char Type)
{
#ifdef R__SSL
   return (XrdCryptoLite *)(new XrdCryptoLite_bf32(Type));
#else
   return (XrdCryptoLite *)0;
#endif
}
