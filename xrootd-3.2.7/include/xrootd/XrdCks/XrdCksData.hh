#ifndef __XRDCKSDATA_HH__
#define __XRDCKSDATA_HH__
/******************************************************************************/
/*                                                                            */
/*                         X r d C k s D a t a . h h                          */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <string.h>
  
class XrdCksData
{
public:

static const int NameSize = 16; // Max name  length is NameSize - 1
static const int ValuSize = 64; // Max value length is 512 bits

char      Name[NameSize];       // Checksum algorithm name
long long fmTime;               // File's mtime when checksum was computed.
int       csTime;               // Delta from fmTime when checksum was computed.
short     Rsvd1;                // Reserved field
char      Rsvd2;                // Reserved field
char      Length;               // Length, in bytes, of the checksum value
char      Value[ValuSize];      // The binary checksum value

int       Get(char *Buff, int Blen)
             {const char *hv = "0123456789abcdef";
              int i, j = 0;
              if (Blen < Length*2+1) return 0;
              for (i = 0; i < Length; i++)
                  {Buff[j++] = hv[(Value[i] >> 4) & 0x0f];
                   Buff[j++] = hv[ Value[i]       & 0x0f];
                  }
              Buff[j] = '\0';
              return Length*2;
             }

int       Set(const char *csName)
             {if (strlen(csName) >= sizeof(Name)) return 0;
              strncpy(Name, csName, sizeof(Name));
              return 1;
             }

int       Set(const char *csVal, int csLen)
             {int n, i = 0, Odd = 0;
              if (csLen > (int)sizeof(Value)*2 || (csLen & 1)) return 0;
              Length = csLen/2;
              while(csLen--)
                   {     if (*csVal >= '0' && *csVal <= '9') n = *csVal-48;
                    else if (*csVal >= 'a' && *csVal <= 'f') n = *csVal-87;
                    else if (*csVal >= 'A' && *csVal <= 'F') n = *csVal-55;
                    else return 0;
                    if (Odd) Value[i++] |= n;
                       else  Value[i  ]  = n << 4;
                    csVal++; Odd = ~Odd;
                   }
              return 1;
             }

          XrdCksData() : Rsvd1(0), Rsvd2(0), Length(0)
                       {memset(Name, 0, sizeof(Name));
                        memset(Value,0, sizeof(Value));
                       }
};
#endif
