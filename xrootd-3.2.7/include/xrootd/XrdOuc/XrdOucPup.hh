#ifndef __XRDOUCPUP_HH__
#define __XRDOUCPUP_HH__
/******************************************************************************/
/*                                                                            */
/*                          X r d O u c P u p . h h                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

#include <stddef.h>
#include <sys/types.h>
#include <sys/stat.h>
  
class  XrdSysError;
struct iovec;

/******************************************************************************/
/*                            P a r a m e t e r s                             */
/******************************************************************************/
  
enum XrdOucPupType
{//  PT_Convert = 0x00, // Convert based on the below (same as char here)
     PT_Ignore  = 0x01, // Both: Skip the pup entry
     PT_Mark    = 0x02, // Pack: Mark &iov element in *Data
     PT_Skip    = 0x03, // Pack: Skip  iov element
     PT_MandS   = 0x04, // Pack: Mark  and Skip
     PT_Fence   = 0x05, // Unpk: Remaining entries are optional
     PT_Datlen  = 0x06, // Unpk: Set prv unpacked data length as an int
     PT_Totlen  = 0x07, // Pack: Set total packed data length as an int
     PT_End     = 0x0e, // Both: End of list (always the last element)
     PT_EndFill = 0x0f, // Both: End of list (always the last element)
                        // Pack: *(Base+Doffs) = totlen as net short

     PT_char    = 0x00, // Both: Character
     PT_short   = 0x80, // Both: Unsigned
     PT_int     = 0xa0, // Both: Unsigned
     PT_longlong= 0xc0, // Both: Unsigned
     PT_special = 0xe0, // Both: Reserved

     PT_Inline  = 0x10, // Internal use
     PT_MaskB   = 0x60, // Internal Use
     PT_MaskT   = 0xe0, // Internal Use
     PT_Mask    = 0xf0, // Internal Use
     PT_MaskD   = 0x0f  // Internal Use
};

struct  XrdOucPupArgs
{       int                 Doffs;     // Offset(data source or target)
        short               Dlen;      // If (Dlen < 0) Dlen = strlen(Data)+1;
        unsigned char       Name;      // Name index of this element
        unsigned char       Dtype;     // One of XrdOucPupType
};

struct XrdOucPupNames
{      const char         **NList;     // -> Array of name pointers
       int                  NLnum;     // Number of elements in NList

       XrdOucPupNames(const char **nlist=0, int nlnum=0)
                     {NList = nlist; NLnum = nlnum;}
      ~XrdOucPupNames() {}
};

#define setPUP0(Type) {0, -1, 0, PT_ ## Type}

#define setPUP1(Name,Type,Base,Var) \
               {offsetof(Base,Var),   -1, Name, PT_ ## Type}

#define setPUP2(Name,Type,Base,Var,Dlen) \
               {offsetof(Base,Var), Dlen, Name, PT_ ## Type}

/******************************************************************************/
/*                             X r d O u c P u p                              */
/******************************************************************************/
  
class XrdOucPup
{
public:

static const int MaxLen = 0x7ff;

// Pack #1: Packs a true null terminated character string. The result is placed
//          in iovec which must have at least two elements. Always returns the
//          length of the packed result with iovec updated to point to the
//          next free element.
//
static int   Pack(struct iovec **, const char *, unsigned short &buff);

// Pack #2: Packs a binary stream of length dlen. The result is placed
//          in iovec which must have at least two elements. Always returns the
//          length of the packed result with iovec updated to point to the
//          next free element.
//
static int   Pack(struct iovec **, const char *, unsigned short &, int dlen);

// Pack #3: Packs an int into buff and returns the length of the result. The
//          pointer to buff is updated to point to the next free byte. The
//          sizeof(buff) must be at least sizeof(int)+1.
//
static int   Pack(char **buff, unsigned int data);

// Pack #4: Packs a binary stream of length dlen when dlen >= 0; Otherwise,
//          it packs a string where dlen is strlen(data)+1. The results is
//          placed in buff which must be atleast dlen+2 long. It returns the
//          length of the packed result with buff updated to point to the
//          next free byte.
//
static int   Pack(char **buff, const char *data, int dlen=-1);

// Pack #5: Packs arbitrary data as directed by XrdOucPupArgs. Data comes from
//          an area pointed to by (Base+PupArgs.Doffs). The reults is placed in
//          iovec (1st arg). The 2nd iovec arg points to the last element+1 and
//          is used to check for an over-run. The Work buffer is used to hold
//          interleaved meta-data and should be sized 9*number of conversions.
//          Returns the actual number of elements used or zero upon an error.
//
       int   Pack(struct iovec *, struct iovec *, XrdOucPupArgs *,
                  char *Base,     char *Work);

// Unpack #1: Unpacks a character or binary string in buff bounded by bend.
//            The pointer to the string is placed in data and it length in dlen.
//            Returns true upon success and false upon failure.
//
static int   Unpack(char **buff, const char *bend, char **data, int &dlen);

// Unpack #2: Unpacks an arbitrary stream in buff bounded by bend as directed
//            by pup. Arg Base is the address of the buffer where data is to be
//            placed as directed by (base+pup->Doffs). All variables in this
//            buffer (e.g., pointers, ints, etc) must be properly aligned.
//            Returns true upon success and false otherwise.
//
       int   Unpack(const char *buff, const char *bend, XrdOucPupArgs *pup,
                          char *base);

       XrdOucPup(XrdSysError *erp=0, XrdOucPupNames *nms=0)
                {eDest = erp, Names = nms;}
      ~XrdOucPup() {}

private:
       int eMsg(const char *etxt, int ino, XrdOucPupArgs *pup);

       XrdSysError    *eDest;
       XrdOucPupNames *Names;
};
#endif
