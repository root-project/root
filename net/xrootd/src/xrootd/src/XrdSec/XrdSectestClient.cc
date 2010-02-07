/******************************************************************************/
/*                                                                            */
/*                   X r d S e c t e s t C l i e n t . c c                    */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdSectestClientCVSID = "$Id$";

/* Syntax: testClient [-b] [-d] [-h host] [-l] [sectoken]

   See the help() function for an explanation of the above.
*/
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/param.h>
#include <sys/socket.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSec/XrdSecInterface.hh"
  
/******************************************************************************/
/*                    L O C A L   D E F I N I T I O N S                       */
/******************************************************************************/

#define H(x)         fprintf(stderr,x); fprintf(stderr, "\n");
#define I(x)         fprintf(stderr, "\n"); H(x)

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char **argv)
{
char *tohex(char *inbuff, int inlen, char *outbuff);

char *protocols=0, *hostspec=0;

struct sockaddr_in netaddr;
netaddr.sin_family = AF_INET;
netaddr.sin_port   = 0;
netaddr.sin_addr.s_addr = 0x80000001;

int putbin = 0, putlen = 0;
char kbuff[8192];
char c;

XrdSecCredentials *cred;
XrdSecParameters   SecToken;
XrdSecProtocol    *pp;
int DebugON = 0;
void help(int);


   /*Get all of the options.
    */
    while ((c=getopt(argc,argv,"bdlh:")) != (char)EOF)
      { switch(c)
        {
        case 'b': putbin = 1;                         break;
        case 'd': DebugON = 1;                        break;
        case 'h': hostspec = optarg;                  break;
        case 'l': putlen = 1;                         break;
        default:  help(1);
        }
      }

// Check if the security token is the last argument
//
   if (optind < argc) protocols = argv[optind++];

/*Make sure no more parameters exist.
*/
   if (optind < argc) 
      {cerr <<"testClient: Extraneous parameter, '" <<argv[optind] <<"'." <<endl;
       help(2);
      }

// Determine protocol string
//
   if (!protocols && !(protocols = getenv("XrdSecSECTOKEN")))
      {cerr <<"testClient: Security protocol string not specified." <<endl;
       help(2);
      }
   SecToken.size = strlen(protocols);
   SecToken.buffer = protocols;

// if hostname given, get the hostname address
//
   if (hostspec)
      {struct hostent *hp;
       if (!(hp = gethostbyname(hostspec)))
          {cerr <<"testServer: host '" <<hostspec <<"' not found." <<endl;
           exit(1);
          }
       memcpy((void *)&netaddr.sin_addr.s_addr, hp->h_addr_list[0],
              sizeof(netaddr.sin_addr.s_addr));
      } else hostspec = (char *)"localhost";

// Do debug processing
//
   if (DebugON)
      {putenv((char *)"XrdSecDEBUG=1");
       cerr <<"testClient: security token='" <<protocols <<"'" <<endl;
      }

// Get the protocol
//
   pp = XrdSecGetProtocol(hostspec, (const struct sockaddr &)netaddr,SecToken,0);
   if (!pp) {cerr << "Unable to get protocol." <<endl; exit(1);}

// Get credentials using this context
//
   cred = pp->getCredentials();
   if (!cred)
      {cerr << "Unable to get credentials," <<endl;
       exit(1);
      }
   if (DebugON)
      cerr << "testClient: credentials size=" <<cred->size <<endl;

// Write out the credentials
//
   if (putbin)
      {if (putlen)
          {if (fwrite(&cred->size, sizeof(cred->size), 1, stdout) != sizeof(cred->size))
	      {cerr << "Unable to write credentials length" <<endl; 
	       exit(1);}}
       if (fwrite((char *) cred->buffer, cred->size, 1, stdout) != (size_t) cred->size)
          {cerr << "Unable to write credentials" <<endl; 
           exit(1);}
      } else {
       if (putlen) printf("%s",
                tohex((char *)&cred->size, sizeof(cred->size), kbuff));
       printf("%s\n", tohex((char *) cred->buffer, cred->size, kbuff));
      }

// All done.
//
   pp->Delete();
}

char *tohex(char *inbuff, int inlen, char *outbuff) {
     static char hv[] = "0123456789abcdef";
     int i, j = 0;
     for (i = 0; i < inlen; i++) {
         outbuff[j++] = hv[(inbuff[i] >> 4) & 0x0f];
         outbuff[j++] = hv[ inbuff[i]       & 0x0f];
         }
     outbuff[j] = '\0';
     return outbuff;
     }

/*help prints hout the obvious.
*/
void help(int rc) {
/* Use H macro to avoid Sun string catenation bug. */
I("Syntax:   testClient [ options ] [sectoken]")
I("Options:  -b -d -l -h host")
I("Function: Request for credentials relative to an operation.")

if (rc > 1) exit(rc);
I("options:  (defaults: -o 01")
I("-b        output the ticket in binary format (i.e., not hexchar).")
I("-d        turns on debugging.")
I("-l        prefixes the ticket with its 4-byte length.")
I("-h host   the requesting hostname (default is localhost).")
I("Notes:    1. Variable XrdSecSECTOKEN must contain the security token,")
H("             sectoken, if it is not specified on the command line.")
exit(rc);
}
