/******************************************************************************/
/*                                                                            */
/*                         X r d O u c A r g s . c c                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOucArgsCVSID = "$Id$";

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>

#include "XrdOuc/XrdOucArgs.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*             L o c a l   C l a s s   X r d O u c A r g s O p t              */
/******************************************************************************/
  
class XrdOucArgsXO
{
public:

int   operator==(char *optarg)
        {int i = strlen(optarg);
         return i <= Optmaxl && i >= Optminl &&
                !strncmp((const char *)Optword, optarg, i);
        }

char *operator%(char *optarg)
        {int i = strlen(optarg);
         XrdOucArgsXO *p = this;
         do if (i <= p->Optmaxl && i >= p->Optminl &&
               !strncmp((const char *)p->Optword, optarg, i)) return p->Optvalu;
            while((p = p->Optnext));
         return 0;
        }

      XrdOucArgsXO(XrdOucArgsXO *nP,const char *optw,int minl,const char *optm)
              {Optword = strdup(optw);
               Optminl = minl; Optmaxl = strlen(optw);
               Optvalu[0] = optm[0];
               Optvalu[1] = (optm[0] ? optm[1] : '\0');
               Optnext = nP;
              }

     ~XrdOucArgsXO()
              {if (Optword) free(Optword);
               if (Optnext) delete Optnext;
              }
private:
XrdOucArgsXO  *Optnext;
char          *Optword;
int            Optmaxl;
int            Optminl;
char           Optvalu[2];
};
  
/******************************************************************************/
/*              C o n s t r u c t o r   &   D e s t r u c t o r               */
/******************************************************************************/
  
XrdOucArgs::XrdOucArgs(XrdSysError *erp,
                       const char  *etxt,
                       const char  *StdOpts,
                       const char  *optw,
                    // int          minl,
                    // const char  *optm,
                                    ...) : arg_stream(0)
{
   va_list ap;
   const char *optm;
   int minl;

// Do the standard initialization
//
   inStream = Argc = Aloc = 0; vopts = curopt = 0; endopts = 1;
   optp = 0; eDest = erp;
   epfx = strdup(etxt ? etxt : "");

// Process teh valid opts
//
   if (StdOpts && *StdOpts == ':') {missarg = ':'; StdOpts++;}
      else missarg = '?';
   vopts = strdup(StdOpts ? StdOpts : "");

// Handle list of extended options, if any
//
   if (optw)
      {va_start(ap, optw);
       while(optw)
            {minl = va_arg(ap, int);
             optm = va_arg(ap, const char *);
             optp = new XrdOucArgsXO(optp, optw, minl, optm);
             optw = va_arg(ap, const char *);
            }
       va_end(ap);
      }
}

/******************************************************************************/

XrdOucArgs::~XrdOucArgs()
             {if (vopts) free(vopts);
              if (optp) delete optp;
              free(epfx);
             }

/******************************************************************************/
/*                               g e t a r g s                                */
/******************************************************************************/
  
char *XrdOucArgs::getarg()
{

// Return argument from whatever source we have
//
   if (inStream) return arg_stream.GetToken();
   if (Aloc >= Argc) return (char *)0;
   argval = Argv[Aloc++];
   return argval;
}

/******************************************************************************/
/*                                g e t o p t                                 */
/******************************************************************************/
  
char XrdOucArgs::getopt()
{
   char optbuff[3] = {'-', 'x', '\0'}, *optspec, *arglist, *optname = 0;

// Check if we really have any more options
//
   if (endopts) return -1;

// Get next option from whatever source we have
//
   if (curopt && *curopt) curopt++;
      else if (inStream)
              {if ((optname = curopt = arg_stream.GetToken(&arglist)))
                  {if (*curopt != '-') {arg_stream.RetToken(); curopt = 0;}
                      else curopt++;
                  }
              }
              else if (Aloc >= Argc || *Argv[Aloc] != '-') curopt = 0;
                      else optname = curopt = Argv[Aloc++]+1;

// Check if we really have an option here
//
   if (!curopt) {endopts = 1; return -1;}
   if (!*curopt)
      {if (eDest) eDest->Say(epfx, "Option letter missing after '-'.");
       endopts = 1;
       return '?';
      }

// Check for extended options or single letter option
//
   if (*curopt == ':' || *curopt == '.') optspec = 0;
      else {if (optp) {optspec = *optp%curopt; curopt = 0;}
               else {optspec = index(vopts, int(*curopt));
                     optbuff[1] = *curopt; optname = optbuff; curopt++;
                    }
           }
   if (!optspec)
      {char buff[500];
       if (eDest)
          {sprintf(buff, "Invalid option, '%s'.", optname);
           eDest->Say(epfx, buff);
          }
       endopts = 1;
       return '?';
      }

// Check if this option requires an argument
//
   if (optspec[1] != ':' && optspec[1] != '.') return *optspec;

// Get the argument from whatever source we have
//
   if (inStream) argval = arg_stream.GetToken();
      else argval = (Aloc < Argc ? Argv[Aloc++] : 0);

// If we have a valid argument, then we are all done
//
   if (argval)
      {if (!*argval) argval = 0;
         else if (*argval != '-') return *optspec;
      }

// If argument is optional, let it go
//
   if (optspec[1] == '.')
      {if (argval && *argval == '-')
          {if (inStream) arg_stream.RetToken();
             else Aloc--;
          }
       argval = 0;
       return *optspec;
      }

// Complain about a missing argument
//
   if (eDest) eDest->Say(epfx, "Value not specified for '", optname, "'.");
   endopts = 1;
   return missarg;
}
  
/******************************************************************************/
/*                                   S e t                                    */
/******************************************************************************/

void XrdOucArgs::Set(char *arglist)
{
   inStream = 1; 
   arg_stream.Attach(arglist);
   curopt = 0;
   endopts = !arg_stream.GetLine();
}

void XrdOucArgs::Set(int argc, char **argv)
{
   inStream = 0; 
   Argc = argc; Argv = argv; Aloc = 0;
   curopt = 0; endopts = 0;
   endopts = !argc;
}
