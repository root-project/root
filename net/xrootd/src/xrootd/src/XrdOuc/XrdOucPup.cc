/******************************************************************************/
/*                                                                            */
/*                          X r d O u c P u p . c c                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOucPupCVSID = "$Id$";

#include <errno.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#ifdef WIN32
#include <direct.h>
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdOuc/XrdOucPup.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
 
/******************************************************************************/
/*                                  P a c k                                   */
/******************************************************************************/

int XrdOucPup::Pack(struct iovec  **iovP,
                    const char     *data,
                    unsigned short &buff)
{
   unsigned short dlen;
   struct iovec *vP = *iovP;

// Pack the data as "<short len><char>" if data is present or "<short 0>" o/w.
// When data is present, the null byte is always included.
//
   vP->iov_base = (char *)&buff; vP->iov_len = sizeof(buff); vP++;

   if (data)
      {dlen = static_cast<unsigned short>(strlen(data)+1);
       buff = htons(dlen);
       vP->iov_base = (char *)data; vP->iov_len = dlen; vP++;
      } else {buff = 0; dlen = 0;}

   *iovP = vP;
   return dlen+sizeof(buff);
}

/******************************************************************************/

int XrdOucPup::Pack(struct iovec  **iovP,
                    const char     *data,
                    unsigned short &buff,
                    int             dlen)
{
   struct iovec *vP = *iovP;

   vP->iov_base = (char *)&buff; vP->iov_len = sizeof(buff); vP++;

   if (data)
      {buff = htons(static_cast<unsigned short>(dlen));
       vP->iov_base = (char *)data; vP->iov_len = dlen; vP++;
      } else {buff = 0; dlen = 0;}

   *iovP = vP;
   return dlen+sizeof(buff);
}

/******************************************************************************/

int XrdOucPup::Pack(char          **buff,
                    const char     *data,
                    int             dlen)
{
   char *bp = *buff;
   unsigned short xlen;

   if (data)
      {if (dlen < 0) dlen = strlen(data)+1;
       xlen = htons(static_cast<unsigned short>(dlen));
       memcpy(bp, &xlen, sizeof(xlen)); bp += sizeof(xlen);
       memcpy(bp, data, dlen);
       bp += dlen;
      } else {*bp++ = '\0'; *bp++ = '\0'; dlen = 0;}

   *buff = bp;
   return dlen+sizeof(xlen);
}

/******************************************************************************/

int XrdOucPup::Pack(char          **buff,
                    unsigned int    data)
{
   unsigned int netData = htonl(data);
   char *bp = *buff;

   if (netData & PT_Mask)
      {*bp = static_cast<char>(PT_int);
       memcpy(bp+1, &netData, sizeof(netData));
       *buff = bp + sizeof(netData)+1;
       return sizeof(netData)+1;
      }
   (*(char *)&netData) |= PT_int | PT_Inline;
   memcpy(bp, &netData, sizeof(netData));
   *buff = bp + sizeof(netData);
   return sizeof(netData);
}
  
/******************************************************************************/

int XrdOucPup::Pack(struct iovec *iovP, struct iovec *iovE, XrdOucPupArgs *pup,
                           char  *base,        char  *Work)
{
   static char Nil[] = {PT_char, '\0'};
   static const int Sz16 = sizeof(short)       + 1;
   static const int Sz32 = sizeof(int)         + 1;
   static const int Sz64 = sizeof(long long)   + 1;
   struct iovec  *vP = iovP;
   XrdOucPupArgs *pP = pup;
   char          *wP = Work;
   int Dtype, dlen = 0, TotLen = 0;

          unsigned long long  n64;
          unsigned int        n32;
          unsigned short      n16;

   union {unsigned long long *B64;
          unsigned int       *B32;
          unsigned short     *B16;
          char              **B08;} Base;

   do {Base.B08 = (char **)(base + pP->Doffs);
       Dtype = pP->Dtype;
       //cerr <<"arg " <<pP-pup <<" type " <<Dtype <<' '
       //     <<(Names->NList[pP->Name] ? Names->NList[pP->Name] : "?") <<endl;
       switch(Dtype)
             {case PT_char:
/* Null Pointer */ if (!*Base.B08) {vP->iov_base = Nil; vP->iov_len  = 2;
                                   vP++; TotLen += 2; break;
                                  }
                   dlen = (pP->Dlen < 0 ? strlen(*Base.B08)+1 : pP->Dlen);
/* Full String */  if (dlen > MaxLen)
                      return eMsg("string too long packing", int(pP-pup), pP);
                   if (vP >= iovE)
                      return eMsg("too many args packing", int(pP-pup), pP);
                   n16 = htons(static_cast<unsigned short>(dlen));
                   vP->iov_base = wP;        vP->iov_len  =  sizeof(n16); vP++;
                   memcpy(wP, &n16, sizeof(n16)); wP += sizeof(n16);
                   vP->iov_base = *Base.B08; vP->iov_len  =  dlen;        vP++;
                   TotLen += dlen + sizeof(n16);
                   break;

              case PT_short:
                   n16 = htons(*Base.B16);
                   *wP = static_cast<char>(PT_short);
                   memcpy(wP+1, &n16, sizeof(n16));
                   vP->iov_base = wP; vP->iov_len = Sz16; vP++;
                   wP += Sz16; TotLen += Sz16; dlen = sizeof(n16);
                   break;

              case PT_int:
                   n32 = htonl(*Base.B32);
                   *wP = static_cast<char>(PT_int);
                   memcpy(wP+1, &n32, sizeof(n32));
                   vP->iov_base = wP; vP->iov_len = Sz32; vP++;
                   wP += Sz32; TotLen += Sz32; dlen = sizeof(n32);
                   break;

              case PT_longlong:
                         h2nll(*Base.B64, n64);
                   *wP = static_cast<char>(PT_longlong);
                   memcpy(wP+1, &n64, sizeof(n64));
                   vP->iov_base = wP; vP->iov_len = Sz64; vP++;
                   wP += Sz64; TotLen += Sz64; dlen = sizeof(n64);
                   break;

              case PT_special: break;

              case PT_Fence:   break;
              case PT_Ignore:  break;

              case PT_MandS:
              case PT_Mark:
                   *Base.B08 = (char *)vP;
                   if (pP->Dtype == PT_Mark) break;

              case PT_Skip:
                   vP++;
                   break;

              case PT_Datlen:
                   *Base.B32 = dlen;
                   break;

              case PT_Totlen:
                   *Base.B32 = TotLen;
                   break;

              case PT_EndFill: 
                   *Base.B16 = htons(static_cast<unsigned short>(TotLen));
              case PT_End:     
                   return static_cast<int>(vP-iovP);

              default: {}
             }
       pP++;
      } while(vP < iovE);

// We over-ran the iovec array
//
   return eMsg("arg list too long packing", int(pP-pup), pup);
}

/******************************************************************************/
/*                                U n p a c k                                 */
/******************************************************************************/
  

int XrdOucPup::Unpack(      char **buff,
                      const char  *bend,
                            char **data,
                            int   &dlen)
{
   unsigned short temp;
            char *bnxt = *buff;

// Grab the length but make sure it's within bounds
//
   if ((bnxt = bnxt+sizeof(temp)) > bend) return 0;
   memcpy(&temp, *buff, sizeof(temp));
   dlen  = static_cast<int>(ntohs(temp));

// Now grab the data
//
   if (dlen) {*data = (char *)bnxt; bnxt += dlen;}
      else    *data = 0;
   *buff = bnxt;
   return (bnxt <= bend);
}

/******************************************************************************/

int XrdOucPup::Unpack(const char    *buff, const char *bend,
                      XrdOucPupArgs *pup,        char *base)
{
   const char *bp = buff, *dp;
   XrdOucPupArgs *uP = pup;
   int dlen = 0, Dtype, Aok = 0, Done = 0;
   union {unsigned long long  b64;
          unsigned int        b32;
          unsigned short      b16;
          unsigned char       b08;} Temp;

   union {unsigned long long *B64;
          unsigned int       *B32;
          unsigned short     *B16;
          char              **B08;} Base;


   while(!Done)
        {Base.B08 = (char **)(base+uP->Doffs);
         if (uP->Dtype & PT_MaskD)
            {switch(uP->Dtype)
                   {case PT_Fence:   Aok = 1;         break;
                    case PT_Datlen: *Base.B32 = dlen; break;
                    case PT_End:
                    case PT_EndFill: Done = 1; uP--;  break;
                    default: {}
                   }
             uP++; continue;
            }
         if (bp+2 > bend)
            return eMsg("buffer overrun unpacking", int(uP-pup), uP);
         if (uP->Dtype == PT_char && !(*bp & PT_short))
            {memcpy(&Temp.b16, bp, sizeof(unsigned short));
             dlen = static_cast<int>(ntohs(Temp.b16));
             bp += sizeof(unsigned short);
             if (dlen)
                if (bp+dlen > bend)
                   return eMsg("buffer overrun unpacking", int(uP-pup), uP);
                   else *Base.B08 = (char *)bp;
                else if (!Aok) break;
                        else *Base.B08 = 0;
            } else {
             Dtype = static_cast<int>(*bp & PT_MaskT);
             if ((unsigned char)Dtype != uP->Dtype)
                return eMsg("arg/data mismatch unpacking", int(uP-pup), uP);
             if (!(dlen = (*bp & PT_MaskB)>>3)) dlen = sizeof(unsigned short);
             dp = (*bp & PT_Inline ? bp : bp+1);
             if (dp+dlen > bend)
                return eMsg("buffer overrun unpacking", int(uP-pup), uP);
             memcpy(&Temp.b64, dp, dlen);
             if (bp == dp) Temp.b08 &= PT_MaskD;
                else bp++;
             switch(Dtype)
                   {case PT_short:    *Base.B16 = ntohs(Temp.b16);  break;

                    case PT_int:      *Base.B32 = ntohl(Temp.b32);  break;

                    case PT_longlong: *Base.B64 = ntohll(Temp.b64); break;

                    default: {}
                   }
             }
        uP++; bp += dlen;
       };

// Make sure we are not missing any items
//
   if (Aok || uP->Dtype == PT_End || uP->Dtype == PT_EndFill)
      return static_cast<int>(uP-pup);
   return eMsg("missing arg unpacking", int(uP-pup), uP);
}

/******************************************************************************/
/*                                  e M s g                                   */
/******************************************************************************/
  
int XrdOucPup::eMsg(const char *etxt, int ino, XrdOucPupArgs *pup)
{
   const char *dtn;
   char buff[1024];

// Check if we can print an error message
//
   if (!eDest) return 0;

// Get type name
//
   switch(pup->Dtype)
         {case PT_char:     dtn = "char";      break;
          case PT_short:    dtn = "short";     break;
          case PT_int:      dtn = "int";       break;
          case PT_longlong: dtn = "long long"; break;
          case PT_special:  dtn = "special";   break;
          default:          dtn = "";          break;
         };

// Format the message
//
   sprintf(buff, "%s arg %d: %s.", dtn, ino,
          (!Names||pup->Name >= Names->NLnum ? "?" : Names->NList[pup->Name]));

   eDest->Emsg("Pup", etxt, buff);
   return 0;
}
