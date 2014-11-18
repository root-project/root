#ifndef __XRDSYSFATTR_HH__
#define __XRDSYSFATTR_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d S y s F A t t r . h h                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

class XrdSysError;

// XrdSysFAttr provides a portable interface to handle extended file attributes
//
class XrdSysFAttr
{
public:
/* AList  is a structure which defines attribute names and the associated
          size of the attributes value. It is used for Free() and List().
*/
struct AList
      {AList *Next;    // -> next element.
       int    Vlen;    //   The length of the attribute value;
       int    Nlen;    //   The length of the attribute name that follows.
       char   Name[1]; //   Start of the name (size of struct is dynamic)
      };

/* Copy() copies one or more attributes from iPath file to the oPath file.
          The first form without Aname copies all attributes. The second form
          copies only the attributes pointed to by Aname.
          Success: True
          Failure: False
*/
static int Copy(const char *iPath, int iFD, const char *oPath, int oFD);

static int Copy(const char *iPath, int iFD, const char *oPath, int oFD,
                const char *Aname);

/* Del() removes attribute "Aname" from the file identified by "Path" or
         an opened file referenced by "fd".
         Success:   zero is returned.
         Failure: -errno is returned. Note that no error is returned should
                  "Aname" not exist.
*/
static int Del(const char *Aname, const char *Path, int fd=-1);

/* Free() releases the AList list returned my List(). This method must be
          used to deallocate the storage as AList is dynamically sized.
*/
static void Free(AList *aPL);

/* Get() get the value associated with attribute "Aname" from the file
         identified by "Path" or an opened file referenced by "fd". The value
         is placed in the buffer pointed to by "Aval" whose size if "Avsz"
         bytes. Only up to "Avsz" bytes are returned and no check is made
         to see if more bytes can be returned. To see how many bytes are
         occupied by the attribute value, call Get() with "Avsz" set to zero.
         Success: the number of bytes placed in "Aval" is returned. If
                  "Avsz" is zero, this is how many bytes could have been set.
         Failure: -errno is returned. Should "Aname" not exist then zero is
                  returned (i.e., no value bytes).
*/
static int Get(const char *Aname, void *Aval, int Avsz,
               const char *Path,  int fd=-1);

/* List() returns the list of extended attribute along with the size of each for
          the file identified by "Path" or an opened file referenced by "fd".
          The first element of the list is returned in aPL. You must use the
          class defined Free() method to deallocate the list. If getSZ == True
          then the size of the attribute value is also returned; otherwise,
          the size is set to zero and no maximum size can be returned.
          Success: the length of the lagest attribute value is returned (if
                   getSZ is true; otherwise zero is returned) and
                   *aPL is set to point to the first AList element, if any.
          Failure: -error is returned and *aPL is set to zero.
*/
static int List(AList **aPL, const char *Path, int fd=-1, int getSz=0);

/* Set() sets the value associated with attribute "Aname" for the file
         identified by "Path" or an opened file referenced by "fd". The value
         must be in the buffer pointed to by "Aval" and be "Avsz" bytes long.
         Normally, "Aname" is created if it does not exist or its value is
         simply replaced. By setting isNew to one, then an error is returned
         if Aname already exists and it is not replaced.
         Success:   zero is returned.
         Failure: -errno is returned.
*/
static int Set(const char *Aname, const void *Aval, int Avsz,
               const char *Path,  int fd=-1,  int isNew=0);

/* Msg() is used to establish the error message object. If it is not
   established, no messages are produced. It returns the previous setting.
*/
static XrdSysError *Msg(XrdSysError *erP)
                       {XrdSysError *orP = Say; Say = erP; return orP;}

protected:

static int Diagnose(const char *Op, const char *Var, const char *Path, int ec);
static AList *getEnt(const char *Path,  int fd, const char *Aname,
                     AList *aP, int *msP);

static XrdSysError *Say;
};
#endif
