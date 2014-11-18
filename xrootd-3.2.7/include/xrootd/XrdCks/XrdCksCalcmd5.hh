#ifndef __XRDCKSCALCMD2_HH__
#define __XRDCKSCALCMD5_HH__
/******************************************************************************/
/*                                                                            */
/*                      X r d C k s C a l c m d 5 . h h                       */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdio.h>

#include "XrdCks/XrdCksCalc.hh"
  
class XrdCksCalcmd5 : public XrdCksCalc
{
public:

char       *Current()
                   {MD5Context saveCTX = myContext;
                    char *md5P = Final();
                    myContext = saveCTX;
                    return (char *)md5P;
                   }

void        Init();

XrdCksCalc *New() {return (XrdCksCalc *)new XrdCksCalcmd5;}

char       *Final();

void        Update(const char *Buff, int BLen)
                  {MD5Update((unsigned char *)Buff,(unsigned)BLen);}

const char *Type(int &csSz) {csSz = sizeof(myDigest); return "md5";}

            XrdCksCalcmd5() {Init();}
           ~XrdCksCalcmd5() {}

private:
  
struct MD5Context
      {unsigned int  buf[4];
       unsigned int  bits[2];
       unsigned char in[64];
      };

MD5Context    myContext;
unsigned char myDigest[16];

void byteReverse(unsigned char *buf, unsigned longs);
void MD5Update(unsigned char const *buf, unsigned int len);

#ifndef ASM_MD5
void MD5Transform(unsigned int buf[4], unsigned int const in[16]);
#endif
};
#endif
