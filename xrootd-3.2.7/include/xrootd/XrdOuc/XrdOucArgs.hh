#ifndef __XRDOUCARGS_HH__
#define __XRDOUCARGS_HH__
/******************************************************************************/
/*                                                                            */
/*                         X r d O u c A r g s . h h                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#include <string.h>

#include "XrdOuc/XrdOucTokenizer.hh"

class XrdOucArgsXO;
class XrdSysError;
  
class XrdOucArgs
{
public:

// getarg() returns arguments, if any, one at a time. It should be called after
//          exhausting the option list via getopt() (i.e., it returns args after
//          the last '-' option in the input). Null is returned if no more
//          arguments remain.
//
char *getarg();

// getopt() works almost exactly like the standard C-library getopt(). Some
//          extensions have been implemented see the constructor. In short:
//          ? -> Invalid option or missing option argument (see below).
//          : -> Missing option arg only when StdOpts starts with a colon.
//          -1-> End of option list (can try getarg() now if so wanted).
//
char  getopt();

// Set()    tells this XrdOucArgs where the options and arguments come from.
//          They may come from a character string or from argument array. This
//          simplifies having a command/interactive tool as a single program.
//          You must call Set() prior to getxxx(). You may use the same object
//          over again by simply calling Set() again.
//
void  Set(char *arglist);

void  Set(int argc, char **argv);

// The StdOpts (which may be null) consist repeated single letters each
// optionally followed by a colon (indicating an argument value is needed)
// or a period, indicating an argument value is optional. If neither then the
// single letter option does not have an argument value. The extended options
// map multiple character words to the single letter equivalent (as above).

// Note that this class is not an exact implementation of getopt(), as follows:
// 1) Single letter streams are not supported. That is, each single letter
//    option must be preceeded by a '-' (i.e., -a -b is NOT equivalent to -ab).
// 2) Multi-character options may be preceeded by a single dash. Most other
//    implementation require a double dash. You can simulate this here by just
//    making all your multi-character options start with a dash.
//
      XrdOucArgs(XrdSysError *erp,      // -> Error Message Object (0->silence)
                 const char  *etxt,     // The error text prefix
                 const char  *StdOpts,  // List of standard 1-character options
                 const char  *optw=0,   // Extended option name (0->end of list)
              // int          minl,     // Minimum abbreviation length
              // const char  *optmap,   // Equivalence with 1-character option
                              ...);     // Repeat last 3 args, as desired.

// Example:
// XrdOucArgs myArgs(0, "", "ab:c.e",
//                          "debug", 1, "d",  // -d, -de, -deb, -debu, -debug
//                          "force", 5, "F",  // -force is valid only!
//                          0);               // No more extended options

// Note: getopt() returns the single letter equivalent for long options. So,
//       'd' is returned when -debug is encountered and 'F' for -force.

     ~XrdOucArgs();

char *argval;

private:

XrdOucTokenizer  arg_stream;
XrdSysError     *eDest;
char            *epfx;
XrdOucArgsXO    *optp;
char            *vopts;
char            *curopt;
int              inStream;
int              endopts;
int              Argc;
int              Aloc;
char           **Argv;
char             missarg;
};
#endif
