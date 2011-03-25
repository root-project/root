#ifndef __XRDOUCXATTR_HH__
#define __XRDOUCXATTR_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c X A t t r . h h                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <string.h>
#include <sys/types.h>

#include "XrdSys/XrdSysFAttr.hh"

/* XrdOucXAttr encapsulates a simple extended attribute variable. The name of
   the object encapsulating the xattr definition is a class template argument.
   A template format is used with defined methods for effciency. This means that
   the template argument object must have five methods:

   int         postGet(int Result) - Formats, if necessary, the attribute value
                                     read into the object T if Result > 0.
                                     Result is -errno if an error occurred o/w
                                     it's the number of bytes read. The method
                                     should normaly return Result as this is
                                     returned to the caller as the final result
                                     of the corresponding XrdOucXAttr::Get().

   T          *preSet(T &x)        - Formats, if necessary, the attribute value
                                     prior to writing it out. If formating is
                                     required, the data members should be copied
                                     into the passed object 'x' and changes made
                                     to the copy with the address of 'x' being
                                     returned. If no changes are needed, simply
                                     return 'this' (the address of yourself).
                                     Data is writen from the area pointed to by
                                     the returned pointer.

   const char *Name()              - Provides the attribute name. All attribute
                                     names are automatically placed in the user
                                     namespace so it should not be qualified.

   int         sizeGet()           - Provides the length of the attr value for
                                     Get(). No more than this number of bytes
                                     are read.

   int         sizeSet()           - Provides the length of the attr value for
                                     Set(). This number of bytes are written.

A sample class would be:

class myXattr
{public:

 char myVal[1024]; // Define data members here

 int         postGet(int Result)
                    {if (Result > 0) {<make changes to yourself>}
                     return Result;
                    }

 myXattr    *preSet(myXattr &outXattr)
                    {setXattr = *this;    // Copy   'this' if changes are needed
                     <change setXattr>
                     return &setXattr;    // Return 'this' if no changes needed
                    }

 const char *Name()    {return "myXattr";}

 int         sizeGet() {return sizeof(myXattr);}

 int         sizeSet() {return strlen(myVal)+1;}

             myXattr() {}
            ~myXattr() {}
};

XrdOucXAttr<myXattr> Foo;
*/

/******************************************************************************/
/*                  T e m p l a t e   X r d O u c X A t t r                   */
/******************************************************************************/
  
template<class T>
class XrdOucXAttr
{
public:

T   Attr; // The attribute value

/* Del() removes this attribute from the file identified by Path or an open
         file with file descriptor of fd (fd must be >= 0).
         Success:  Zero  is returned.
         Failure: -errno is returned.
*/
int Del(const char *Path, int fd=-1)
       {return XrdSysFAttr::Del(Attr.Name(), Path, fd);}

/* Get() get this attribute from the file identified by Path or an open file
         with file descriptor of fd (fd must be >= 0). The attribute values are
         placed in the object as defined by Attr above.
         Success: attribute value length is returned.
         Failure: -errno is returned.
*/
int Get(const char *Path, int fd=-1)
       {return Attr.postGet(XrdSysFAttr::Get(Attr.Name(), &Attr, Attr.sizeGet(),
                                             Path, fd));
       }

/* Set() sets the extended attribute for file identified by Path or an open
         file with file descriptor of fd (fd must be >= 0). The values are
         taken from the object Attr, defined above.
         Success:   zero is returned.
         Failure: -errno is returned.
*/
int Set(const char *Path, int fd=-1)
       {T xA;
        return XrdSysFAttr::Set(Attr.Name(), Attr.preSet(xA), Attr.sizeSet(),
                                Path, fd);
       }

    XrdOucXAttr() {}
   ~XrdOucXAttr() {}
};
#endif
