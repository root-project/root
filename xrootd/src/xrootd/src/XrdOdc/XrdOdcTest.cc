/******************************************************************************/
/*                                                                            */
/*                         X r d O d c T e s t . c c                          */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdOdcTestCVSID = "$Id$";

/* This is the distributed cache client test program.

   test_oodc [options] [configfn]

   options: [-d]

Where:
   -d     Turns on debugging mode.

Notes:
   1.     This program puts up a read and reas command like:

          open {r|w|c} <path>

          and types the response. It exits when a null line is entered;
*/

/******************************************************************************/
/*                         i n c l u d e   f i l e s                          */
/******************************************************************************/
  
#include <unistd.h>
#include <ctype.h>
#include <iostream.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "XrdOdc/XrdOdcFinder.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucErrInfo.hh"

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   XrdSysLogger eLog;
   XrdSysError  eDest(&eLog, "odctest_");
   XrdOdcFinder *Finder = (XrdOdcFinder *)(new XrdOdcFinderRMT(&eLog));
   XrdOucStream Stream;
   XrdOucErrInfo   Resp;
   XrdOucEI        Result;
   int retc, mode;
   char c, *cfn = 0, *lp, *tp;
   extern int optind, opterr, optopt;
   void Usage(int);

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc, argv, "d")) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'd': break;
       default:  cerr <<"Test_oodc: Invalid option, -" <<optopt <<endl;
                 Usage(1);
       }
     }

// Set the configuration file name. Bail if there is none
//
   if (optind < argc) cfn = argv[optind];
      else {cerr <<"Test_oodc: Required config file not specified." <<endl;
            Usage(1);
           }

// Configure the finder
//
   if (!Finder->Configure(cfn)) exit(1);

// Attach standard in to a stream
//
   Stream.Attach(0);

// Now read command and process them
//
   while((lp = Stream.GetLine()) && *lp)
        {if (!(tp = Stream.GetToken())) continue;
         if (strcmp("open", tp))
            {cerr <<"Test_oodc: Invalid command - " <<tp <<endl;
             continue;
            }
         if (!(tp = Stream.GetToken()) || strlen(tp) > 1)
            {cerr <<"Test_oodc: Mode (r | w | c) not specified." <<endl;
             continue;
            }
                  if (*tp == 'r') mode = O_RDONLY;
             else if (*tp == 'w') mode = O_RDWR;
             else if (*tp == 'c') mode = O_RDWR | O_CREAT;
             else {cerr <<"Test_oodc: Invalid mode - " <<tp <<endl; continue;}

         if (!(tp = Stream.GetToken()))
            {cerr <<"Test_oodc: Path not specified." <<endl;
             continue;
            }

         retc = Finder->Locate(Resp, tp, mode);
         Resp.getErrInfo(Result);
         cerr <<"Finder result: retc=" <<retc <<" ecode=" <<Result.code <<" resp:" <<endl;
         cerr <<Result.message <<endl;
        }

// If we ever get here, just exit
//
   _exit(0);
}

void Usage(int rc)
     {cerr <<"Usage: test_odc [-d] configfn" <<endl; exit(rc);}
