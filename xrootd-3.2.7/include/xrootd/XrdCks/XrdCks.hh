#ifndef __XRDCKS_HH__
#define __XRDCKS_HH__
/******************************************************************************/
/*                                                                            */
/*                             X r d C k s . h h                              */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdCks/XrdCksData.hh"

class XrdCks;
class XrdOucStream;
class XrdSysError;
class XrdSysPlugin;

/* This class defines the checksum management interface. It should be used as
   the base class for a plugin. When used that way, the shared library holding
   the plugin must define a "C" entry point named XrdCksInit() as described at
   the end of this include file. Note that you can also base you plugin on the
   native implementation, XrdCks, and replace only selected methods.
*/
  
class XrdCks
{
public:

/* Calc()  Calculates a new checksum for the Pfn using the checksum name in
           Cks parameter. The calculated value is returned in Cks as well. If
           doSet is true, the new value replaces any existing value in the
           Pfn's extended attributes.
           Success:  zero is returned.
           Failure: -errno (see significant error numbers below).
*/
virtual
int        Calc( const char *Pfn, XrdCksData &Cks, int doSet=1) = 0;

/* Del()   deletes the checksum from the Pfn's xattrs.
           Success: 0
           Failure: -errno (see significant error numbers below).
*/
virtual
int        Del(  const char *Pfn, XrdCksData &Cks) = 0;

/* Get()   retreives the checksum from the Pfn's xattrs and returns it and
           indicates whether or not it is stale (i.e. the file modification has
           changed or the name and length are not the expected values).
           Success: The length of the binary checksum is returned.
           Failure: -errno (see significant error numbers below).
*/
virtual
int        Get(  const char *Pfn, XrdCksData &Cks) = 0;

/* Config  Is used to parse configuration directives specific to the manager.
           Token points to the directive that triggered the call. Line are all
           the characters after the directive.
           Success:  1
           Failure:  0
*/
virtual
int        Config(const char *Token, char *Line) = 0;

/* Init()  Init is used to fully initialize the manager which includes loading
           any plugins. You can also specify the default checksum. If not given
           it becomes adler32. A default is only needed if you will not be
           specifying the checksum name in the XrdCksData object.
           Success:  1
           Failure:  0
*/
virtual
int        Init(const char *ConfigFN, const char *DfltCalc=0) = 0;

/* List()  returns the names of the checksums associated with Pfn. If Pfn is
           not given, it returns a list of supported checksums. The buffer
           should be at least 64 bytes in length; otherwise truncation occurs.
           Success: Buff is returned with at least one checksum name.
           Failure: A nil pointer is returned.
*/
virtual
char      *List(const char *Pfn, char *Buff, int Blen, char Sep=' ') = 0;

/* Name()  returns the name of the checksums associated with a sequence number.
           Zero is the default name. Higher numbers are alternates. When no
           more alternates exist, a null pointer is returned. Note that Name()
           may be called prior to final config to see if there are any chksums
           to configure and avoid unintended errors.
           Success: Pointer to the name.
           Failure: A nil pointer is returned.
*/
virtual const
char      *Name(int seqNum=0) = 0;

/* Size()  returns the binary length of the checksum with the corresponding
           name. If no name is given, the default name is used.
           Success: checksum length.
           Failure: Zero  if the checksum name does not exist.
*/
virtual
int        Size( const char  *Name=0) = 0;

/* Set()   sets the Pfn's checksum in the extended attributes. The file's mtime
           and the time of setting is automatically added to the information.
           If myTime is true then CksData values for fmTime and gmTime are used.
           Success:  zero is returned.
           Failure: -errno (see significant error numbers below).
*/
virtual
int        Set(  const char *Pfn, XrdCksData &Cks, int myTime=0) = 0;

/* Ver()   retreives the checksum from the Pfn's xattrs and compares it to the
           supplied checksum. If the checksum is not available or is stale,
           a new checksum is calculated and written to the extended attributes.
           Success: True
           Failure: False (the checksums do not match).
                    -errno Otherwise (see significant error numbers below).
*/
virtual
int        Ver(  const char *Pfn, XrdCksData &Cks) = 0;

           XrdCks(XrdSysError *erP) : eDest(erP) {}
virtual   ~XrdCks() {}

/* Significant errno values:

   -EDOM       The supplied checksum length is invalid for the checksum name.
   -ENOTSUP    The supplied or default checksum name is not supported.
   -ESRCH      Checksum does not exist for file.
   -ESTALE     The file's checksum is no longer valid.
*/

protected:

XrdSysError   *eDest;
};

/******************************************************************************/
/*                            X r d C k s I n i t                             */
/******************************************************************************/

#define XRDCKSINITPARMS XrdSysError *, const char *, const char *
  
/* When building a shared library plugin, the following "C" entry point must
   exist in the library:

   extern "C"
          {XrdCks *XrdCksInit(XrdSysError *eDest,  // The error msg object
                              const char  *cFN,    // Config file name
                              const char  *Parms   // Parms via lib directive
                             );
          }

   This entry is called to get an instance of the checksum manager.
   If the object cannot be created; return 0.
*/
#endif
