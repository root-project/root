/******************************************************************************/
/*                                                                            */
/*                        X r d S y s F A t t r . c c                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysFAttr.hh"

/******************************************************************************/
/*                 P l a t f o r m   D e p e n d e n c i e s                  */
/******************************************************************************/

#ifndef ENOATTR
#define ENOATTR ENODATA
#endif
  
/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdSysError *XrdSysFAttr::Say = 0;

/******************************************************************************/
/*                        I m p l e m e n t a t i o n                         */
/******************************************************************************/
  
#if    defined(__FreeBSD__)
#include "XrdSys/XrdSysFAttrBsd.icc"
#elif defined(__linux__)
#include "XrdSys/XrdSysFAttrLnx.icc"
#elif defined(__macos__)
#include "XrdSys/XrdSysFAttrMac.icc"
#elif defined(__solaris__)
#include "XrdSys/XrdSysFAttrSun.icc"
#else
int XrdSysFAttr::Del(const char *Aname, const char *Path)
                {return -ENOTSUP;}
int XrdSysFAttr::Del(const char *Aname, int fd)
                {return -ENOTSUP;}
int XrdSysFAttr::Get(const char *Aname, void *Aval, int Avsz, const char *Path)
                {return -ENOTSUP;}
int XrdSysFAttr::Get(const char *Aname, void *Aval, int Avsz, int fd)
                {return -ENOTSUP;}
int XrdSysFAttr::Set(const char *Aname, const void *Aval, int Avsz,
                     const char *Path,  int isNew)
                {return -ENOTSUP;}
int XrdSysFAttr::Set(const char *Aname, const void *Aval, int Avsz,
                     int         fd,    int isNew)
                {return -ENOTSUP;}
int XrdSysFAttr::Set(XrdSysError *erp) {return 0;}
#endif

/******************************************************************************/
/*                      P r o v i d e d   M e t h o d s                       */
/******************************************************************************/
/******************************************************************************/
/*                                  C o p y                                   */
/******************************************************************************/

int XrdSysFAttr::Copy(const char *iPath, int iFD, const char *oPath, int oFD)
{
   AList *aP, *aNow;
   char *Buff;
   int maxSz;

// Get all of the attributes for the input
//
   if ((maxSz = List(&aP, iPath, iFD, 1)) <= 0)
      return maxSz == 0 || maxSz == -ENOTSUP;

// Allocate a buffer to hold the largest attribute value (plus some)
//
   maxSz += 4096;
   Buff = (char *)malloc(maxSz);

// Get each value and set it
//
   aNow = aP;
   while(aNow && Get(aNow->Name, Buff, maxSz,      iPath, iFD) >= 0
              && Set(aNow->Name, Buff, aNow->Vlen, oPath, oFD) >= 0)
        {aNow = aNow->Next;}

// Free up resources and return
//
   Free(aP);
   free(Buff);
   return aNow == 0;
}

/******************************************************************************/

int XrdSysFAttr::Copy(const char *iPath, int iFD, const char *oPath, int oFD,
                      const char *Aname)
{
   char *bP;
   int sz, rc;

// First obtain the size of the attribute (if zero ignore it)
//
   if ((sz = Get(Aname, 0, 0, iPath, iFD)) <= 0) return (!sz || sz == -ENOTSUP);

// Obtain storage
//
   if (!(bP = (char *)malloc(sz)))
      {Diagnose("get", Aname, oPath, ENOMEM); return 0;}

// Copy over any extended attributes
//
   if ((rc = Get(Aname, bP, sz, iPath, iFD)) > 0)
      {if ((rc = Set(Aname, bP, sz, oPath, oFD)) < 0
       &&  rc == -ENOTSUP) rc = 0;
      }
      else if (rc < 0 && rc == -ENOTSUP) rc = 0;

// All done
//
   free(bP);
   return rc >= 0;
}
  
/******************************************************************************/
/*                              D i a g n o s e                               */
/******************************************************************************/
  
int XrdSysFAttr::Diagnose(const char *Op, const char *Var,
                          const char *Path,  int ec)
{
   char buff[512];

// Screen out common case
//
   if (ec == ENOATTR || ec == ENOENT) return -ENOENT;

// Format message insert and print if we can actually say anything
//
   if (Say)
      {snprintf(buff, sizeof(buff), "%s attr %s from", Op, Var);
       Say->Emsg("FAttr", ec, buff, Path);
      }

// Return negative code
//
   return -ec;
}
  
/******************************************************************************/
/*                     X r d S y s F A t t r : : F r e e                      */
/******************************************************************************/

void XrdSysFAttr::Free(XrdSysFAttr::AList *aLP)
{
   AList *aNP;

// Free all teh structs using free as they were allocated using malloc()
//
   while(aLP) {aNP = aLP->Next; free(aLP); aLP = aNP;}
}

/******************************************************************************/
/*                   X r d S y s F A t t r : : g e t E n t                    */
/******************************************************************************/
  
XrdSysFAttr::AList *XrdSysFAttr::getEnt(const char *Path,  int fd,
                                        const char *Aname,
                                        XrdSysFAttr::AList *aP, int *msP)
{
   AList *aNew;
   int sz = 0, n = strlen(Aname);

// Get the data size of this attribute if so wanted
//
   if (!n || (msP && !(sz = Get(Aname, 0, 0, Path, fd)))) return 0;

// Allocate a new dynamic struct
//
   if (!(aNew = (AList *)malloc(sizeof(AList) + n))) return 0;

// Initialize the structure
//
   aNew->Next = aP;
   aNew->Vlen = sz;
   aNew->Nlen = n;
   strcpy(aNew->Name, Aname); // Gauranteed to fit

// All done
//
   if (msP && *msP < sz) *msP = sz;
   return aNew;
}
