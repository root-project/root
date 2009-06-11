/******************************************************************************/
/*                                                                            */
/*                        X r d S y s E r r o r . c c                         */
/*                                                                            */
/*(c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University   */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*Produced by Andrew Hanushevsky for Stanford University under contract       */
/*           DE-AC03-76-SFO0515 with the Deprtment of Energy                  */
/******************************************************************************/
 
//         $Id$

const char *XrdSysErrorCVSID = "$Id$";

#include <ctype.h>
#ifndef WIN32
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/uio.h>
#else
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define Set_IOV_Item(x, y) {iov[iovpnt].iov_base  = (caddr_t)x;\
                            iov[iovpnt++].iov_len = y;}

#define Set_IOV_Buff(x)    {iov[iovpnt].iov_base  = (caddr_t)x;\
                            iov[iovpnt++].iov_len = strlen(x);}

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdSysError_Table *XrdSysError::etab = 0;

/******************************************************************************/
/*                                b a s e F D                                 */
/******************************************************************************/
  
int XrdSysError::baseFD() {return Logger->originalFD();}

/******************************************************************************/
/*                               e c 2 t e x t                                */
/******************************************************************************/

char *XrdSysError::ec2text(int ecode)
{
    int xcode;
    char *etxt = 0;
    XrdSysError_Table *etp = etab;

    xcode = (ecode < 0 ? -ecode : ecode);
    while((etp != 0) && !(etxt = etp->Lookup(xcode))) etp = etp->next;
    if (!etxt) etxt = strerror(xcode);
    return etxt;
}
  
/******************************************************************************/
/*                                  E m s g                                   */
/******************************************************************************/

int XrdSysError::Emsg(const char *esfx, int ecode, const char *txt1, 
                                                   const char *txt2)
{
    struct iovec iov[16];
    int iovpnt = 0;
    char ebuff[16], etbuff[80], *etxt = 0;

    if (!(etxt = ec2text(ecode)))
       {snprintf(ebuff, sizeof(ebuff), "reason unknown (%d)", ecode); 
        etxt = ebuff;
       } else if (isupper(static_cast<int>(*etxt)))
                 {strlcpy(etbuff, etxt, sizeof(etbuff));
                  *etbuff = static_cast<char>(tolower(static_cast<int>(*etxt)));
                  etxt = etbuff;
                 }

                         Set_IOV_Item(0,0);                          //  0
    if (epfx && epfxlen) Set_IOV_Item(epfx, epfxlen);                //  1
    if (esfx           ) Set_IOV_Buff(esfx);                         //  2
                         Set_IOV_Item(": Unable to ", 12);           //  3
                         Set_IOV_Buff(txt1);                         //  4
    if (txt2 && txt2[0]){Set_IOV_Item(" ", 1);                       //  5
                         Set_IOV_Buff(txt2); }                       //  6
                         Set_IOV_Item("; ", 2);                      //  7
                         Set_IOV_Buff(etxt);                         //  8
                         Set_IOV_Item("\n", 1);                      //  9
    Logger->Put(iovpnt, iov);

    return ecode;
}
  
void XrdSysError::Emsg(const char *esfx, const char *txt1, 
                                         const char *txt2, 
                                         const char *txt3)
{
    struct iovec iov[16];
    int iovpnt = 0;

                         Set_IOV_Item(0,0);                          //  0
    if (epfx && epfxlen) Set_IOV_Item(epfx, epfxlen);                //  1
    if (esfx           ) Set_IOV_Buff(esfx);                         //  2
                         Set_IOV_Item(": ", 2);                      //  3
                         Set_IOV_Buff(txt1);                         //  4
    if (txt2 && txt2[0]){Set_IOV_Item(" ", 1);                       //  5
                         Set_IOV_Buff(txt2);}                        //  6
    if (txt3 && txt3[0]){Set_IOV_Item(" ", 1);                       //  7
                         Set_IOV_Buff(txt3);}                        //  8
                         Set_IOV_Item("\n", 1);                      //  9
    Logger->Put(iovpnt, iov);
}

/******************************************************************************/
/*                                   S a y                                    */
/******************************************************************************/
  
void XrdSysError::Say(const char *txt1, const char *txt2, const char *txt3,
                      const char *txt4, const char *txt5, const char *txt6)
{
    struct iovec iov[9];
    int iovpnt = 0;
    if (txt1)            Set_IOV_Buff(txt1)                          //  0
       else              Set_IOV_Item(0,0);
    if (txt2 && txt2[0]) Set_IOV_Buff(txt2);                         //  1
    if (txt3 && txt3[0]) Set_IOV_Buff(txt3);                         //  2
    if (txt4 && txt4[0]) Set_IOV_Buff(txt4);                         //  3
    if (txt5 && txt5[0]) Set_IOV_Buff(txt5);                         //  4
    if (txt6 && txt6[0]) Set_IOV_Buff(txt6);                         //  5
                         Set_IOV_Item("\n", 1);                      //  6
    Logger->Put(iovpnt, iov);
}
 
/******************************************************************************/
/*                                  T b e g                                   */
/******************************************************************************/
  
void XrdSysError::TBeg(const char *txt1, const char *txt2, const char *txt3)
{
 cerr <<Logger->traceBeg();
 if (txt1) cerr <<txt1 <<' ';
 if (txt2) cerr <<epfx <<txt2 <<": ";
 if (txt3) cerr <<txt3;
}

/******************************************************************************/
/*                                  T E n d                                   */
/******************************************************************************/
  
void XrdSysError::TEnd() {cerr <<endl; Logger->traceEnd();}
