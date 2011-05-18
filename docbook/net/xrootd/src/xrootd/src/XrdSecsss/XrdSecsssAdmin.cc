/******************************************************************************/
/*                                                                            */
/*                     X r d S e c s s s A d m i n . c c                      */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

const char *XrdSecsssAdminCVSID = "$Id$";
  
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysTimer.hh"

#include "XrdSecsss/XrdSecsssKT.hh"
  
/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/
  
#define eMsg(x) cerr <<XrdpgmName <<": " <<x << endl

struct XrdsecsssAdmin_Opts
      {XrdSecsssKT *kTab;
       const char  *Action;
       const char  *KeyName;
       const char  *KeyUser;
       const char  *KeyGrup;
       const char  *KeyFile;
       time_t       Expdt;
       int          Debug;
       int          Keep;
       int          KeyLen;
       int          KeyNum;
       char         Sort;

       XrdsecsssAdmin_Opts() : kTab(0), Action(0), KeyName(0), KeyUser(0),
                               KeyGrup(0), KeyFile(0),
                               Expdt(0), Debug(0), Keep(3), KeyLen(32),
                               KeyNum(-1), Sort('k') {}
      ~XrdsecsssAdmin_Opts() {}
};

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
static const char *XrdpgmName;

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char **argv)
{
   extern char  *optarg;
   extern int    optopt, optind, opterr;
   extern int    XrdSecsssAdmin_addKey(XrdsecsssAdmin_Opts &Opt);
   extern int    XrdSecsssAdmin_delKey(XrdsecsssAdmin_Opts &Opt);
   extern int    XrdSecsssAdmin_insKey(XrdsecsssAdmin_Opts &Opt);
   extern int    XrdSecsssAdmin_lstKey(XrdsecsssAdmin_Opts &Opt);
   extern time_t getXDate(const char *cDate);
   extern void   Usage(int rc, const char *opn=0, const char *opv=0);

   XrdsecsssAdmin_Opts Opt;
   enum What2Do {doAdd, doInst, doDel, doList};
   char c, *sp;
   const char *validOpts = "dg:h:k:l:n:s:u:x:";
   int rc;
   What2Do doIt = doList;

// Get the name of our program
//
   XrdpgmName = ((sp = rindex(argv[0], '/')) ?  sp+1 : argv[0]);

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,validOpts))
         && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'd': Opt.Debug = 1;
                 break;
       case 'g': Opt.KeyGrup = optarg;
                 break;
       case 'h': if ((Opt.Keep   = atoi(optarg)) <= 0) Usage(1, "-s", optarg);
                 break;
       case 'k': Opt.KeyName = optarg;
                 break;
       case 'l': if ((Opt.KeyLen = atoi(optarg)) <= 0 
                 ||   Opt.KeyLen > XrdSecsssKT::ktEnt::maxKLen)
                    Usage(1, "-l", optarg);
                 break;
       case 'n': if ((Opt.KeyNum = atoi(optarg)) <= 0) Usage(1, "-n", optarg);
                 break;
       case 's': if ((int)strlen(optarg) > 1 || !index("cgknux", *optarg))
                    Usage(1, "-s", optarg);
                 Opt.Sort = *optarg;
                 break;
       case 'u': Opt.KeyUser = optarg;
                 break;
       case 'x': if ((Opt.Expdt = getXDate(optarg)) < 0
                 ||   Opt.Expdt < (time(0)+60)) Usage(1, "-x", optarg);
                 break;
       default:  if (index(validOpts, optopt)) Usage(1, argv[optind-1], optarg);
                    else {eMsg("Invalid option '" <<argv[optind-1] <<"'");
                          Usage(1);
                         }
       }
     }

// Make sure and opreration has been specified
//
   if (optind >= argc) {eMsg("Action not specified."); Usage(1);}

// Verify the action
//
        if (!strcmp(argv[optind], "add"))      doIt = doAdd;
   else if (!strcmp(argv[optind], "install"))  doIt = doInst;
   else if (!strcmp(argv[optind], "del"))      doIt = doDel;
   else if (!strcmp(argv[optind], "list"))     doIt = doList;
   else Usage(1, "parameter", argv[optind]);
   Opt.Action = argv[optind++];

// Make sure keyname is not too long
//
   if (Opt.KeyName && (int)strlen(Opt.KeyName) >= XrdSecsssKT::ktEnt::NameSZ)
      {eMsg("Key name must be less than " <<XrdSecsssKT::ktEnt::NameSZ
            << " characters.");
       exit(4);
      }

// Make sure username is not too long
//
   if (Opt.KeyUser && (int)strlen(Opt.KeyUser) >= XrdSecsssKT::ktEnt::UserSZ)
      {eMsg("User name must be less than " <<XrdSecsssKT::ktEnt::UserSZ
            << " characters.");
       exit(4);
      }

// Make sure group name is not too long
//
   if (Opt.KeyGrup && (int)strlen(Opt.KeyGrup) >= XrdSecsssKT::ktEnt::GrupSZ)
      {eMsg("group name must be less than " <<XrdSecsssKT::ktEnt::GrupSZ
            << " characters.");
       exit(4);
      }

// Provide default keyfile if none specified
//
   if (optind < argc) Opt.KeyFile = argv[optind++];
      else            Opt.KeyFile = XrdSecsssKT::genFN();

// Perform the action
//
   switch(doIt)
         {case doAdd:  rc = XrdSecsssAdmin_addKey(Opt); break;
          case doDel:  rc = XrdSecsssAdmin_delKey(Opt); break;
          case doInst: rc = XrdSecsssAdmin_insKey(Opt); break;
          case doList: rc = XrdSecsssAdmin_lstKey(Opt); break;
          default:     rc = 16; eMsg("Internal processing error!");
         }

// All done
//
   if (Opt.kTab) delete Opt.kTab;
   exit(rc);
}

/******************************************************************************/
/*                              g e t X D a t e                               */
/******************************************************************************/

time_t getXDate(const char *cDate)
{
   struct tm myTM;
   char *eP;
   long  theVal;

// if no slashes then this is number of days
//
   if (!index(cDate, '/'))
      {theVal = strtol(cDate, &eP, 10);
       if (errno || *eP) return -1;
       if (theVal) theVal = XrdSysTimer::Midnight() + (86400*theVal);
       return static_cast<time_t>(theVal);
      }

// Do a date conversion
//
   eP = strptime(cDate, "%D", &myTM);
   if (*eP) return -1;
   return mktime(&myTM);
}
  
/******************************************************************************/
/*                                  i s N o                                   */
/******************************************************************************/
  
int isNo(int dflt, const char *Msg1, const char *Msg2, const char *Msg3)
{
   char Answer[8];

   cerr <<XrdpgmName <<": " <<Msg1 <<Msg2 <<Msg3;
   cin.getline(Answer, sizeof(Answer));
   if (!*Answer) return dflt;

   if (!strcmp("y",Answer) || !strcmp("ye",Answer) || !strcmp("yes",Answer))
      return 0;
   return 1;
}

/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
void Usage(int rc, const char *opn, const char *opv)
{
// Check if we need to issue a message here
//
   if (opn)
      {if (opv) eMsg("Invalid " <<opn <<" argument - " <<opv);
          else  eMsg(opn <<" argument not specified.");
      }

cerr <<"\nUsage:   " <<XrdpgmName <<" [options] action\n";
cerr <<"\nOptions: [-d] [-g grpname] [-h hold] [-k keyname] [-l keylen] [-n keynum]";
cerr <<"\n         [-s {c|g|k|n|u|x}] [-u usrname] [-x {days | mm/dd/yy}]" <<endl;
cerr <<"\nActions: {add | del | install | list} [keyfn]" <<endl;
exit(rc);
}

/******************************************************************************/
/*                 X r d S e c s s s A d m i n _ a d d K e y                  */
/******************************************************************************/
  
int  XrdSecsssAdmin_addKey(XrdsecsssAdmin_Opts &Opt)
{
   XrdOucErrInfo eInfo;
   XrdSecsssKT::ktEnt *ktEnt;
   int retc, numKeys, numTot, numExp;

// Allocate the initial keytab
//
   Opt.kTab = new XrdSecsssKT(&eInfo, Opt.KeyFile, XrdSecsssKT::isAdmin);
   if ((retc = eInfo.getErrInfo()))
      {if (retc != ENOENT || isNo(0, "Keyfile '", Opt.KeyFile,
           "' does not exist. Create it? (y | n): ")) return 4;
      }

// Construct a new KeyTab entry
//
   ktEnt = new XrdSecsssKT::ktEnt;
   strcpy(ktEnt->Data.Name, (Opt.KeyName ? Opt.KeyName : "nowhere"));
   strcpy(ktEnt->Data.User, (Opt.KeyUser ? Opt.KeyUser : "nobody"));
   strcpy(ktEnt->Data.Grup, (Opt.KeyGrup ? Opt.KeyGrup : "nogroup"));
        if (Opt.KeyLen > XrdSecsssKT::ktEnt::maxKLen)
           ktEnt->Data.Len = XrdSecsssKT::ktEnt::maxKLen;
   else if (Opt.KeyLen < 4) ktEnt->Data.Len = 4;
   else ktEnt->Data.Len = Opt.KeyLen/4*4;
   ktEnt->Data.Exp = Opt.Expdt;
   Opt.kTab->addKey(*ktEnt);

// Now rewrite the file
//
   if ((retc = Opt.kTab->Rewrite(Opt.Keep, numKeys, numTot, numExp)))
      {eMsg("Unable to add key to '" <<Opt.KeyFile <<"'; " <<strerror(retc));
       retc = 8;
      } else {
       eMsg(numKeys <<(numKeys == 1 ? " key":" keys") <<" out of "
            <<numTot <<" kept (" <<numExp <<" expired).");
      }

// All done
//
   return retc;
}

/******************************************************************************/
/*                 X r d S e c s s s A d m i n _ d e l K e y                  */
/******************************************************************************/
  
int  XrdSecsssAdmin_delKey(XrdsecsssAdmin_Opts &Opt)
{
   XrdOucErrInfo eInfo;
   XrdSecsssKT::ktEnt ktEnt;
   int retc, numKeys, numTot, numExp, numDel;

// Allocate the initial keytab
//
   Opt.kTab = new XrdSecsssKT(&eInfo, Opt.KeyFile, XrdSecsssKT::isAdmin);
   if ((retc = eInfo.getErrInfo()))
      {if (retc == ENOENT) 
          {eMsg("Keyfile '" <<Opt.KeyFile <<"' does not exist.");}
       return 4;
      }

// Construct deletion reference
//
   if (Opt.KeyName) strcpy(ktEnt.Data.Name, Opt.KeyName);
   if (Opt.KeyUser) strcpy(ktEnt.Data.User, Opt.KeyUser);
   if (Opt.KeyGrup) strcpy(ktEnt.Data.Grup, Opt.KeyGrup);
   ktEnt.Data.ID = static_cast<long long>(Opt.KeyNum);

// Delete the keys from the key table
//
   if (!(numDel = Opt.kTab->delKey(ktEnt)))
      {eMsg("No matching key(s) found.");
       return 4;
      }

// It's possible that all of the keys were deleted. Check for that
//
   if (Opt.kTab->keyList() == 0)
      {if (isNo(1, "No keys will remain in ", Opt.KeyFile,
                   ". Delete file? (n | y): "))
          {eMsg("No keys deleted!"); return 2;}
       unlink(Opt.KeyFile);
       return 0;
      }

// Now rewrite the file
//
   if ((retc = Opt.kTab->Rewrite(Opt.Keep, numKeys, numTot, numExp)))
      {eMsg("Unable to del key from '" <<Opt.KeyFile <<"'; " <<strerror(retc));
       retc = 8;
      } else {
       eMsg(numKeys <<(numKeys == 1 ? " key":" keys") <<" out of "
            <<(numTot+numDel) <<" kept (" <<numExp <<" expired).");
      }

// All done
//
   return retc;
}

/******************************************************************************/
/*                 X r d S e c s s s A d m i n _ i n s K e y                  */
/******************************************************************************/
  
int  XrdSecsssAdmin_insKey(XrdsecsssAdmin_Opts &Opt)
{
   extern int XrdSecsssAdmin_isKey(XrdsecsssAdmin_Opts &Opt,
                                   XrdSecsssKT::ktEnt *ktP);
   XrdOucErrInfo eInfo;
   XrdSecsssKT::ktEnt *ktP;
   int retc, numKeys = 0, numTot, numExp;

// Allocate the initial keytab
//
   Opt.kTab = new XrdSecsssKT(&eInfo, 0, XrdSecsssKT::isAdmin);
   if ((retc = eInfo.getErrInfo())) return 4;

// Check if we need to trim the keytab to a particular key
//
   if (Opt.KeyName || Opt.KeyUser || Opt.KeyGrup)
      {ktP = Opt.kTab->keyList();
       while(ktP)
            {if (!XrdSecsssAdmin_isKey(Opt, ktP)) ktP->Data.Name[0] = '\0';
                else numKeys++;
             ktP = ktP->Next;
            }
       if (!numKeys)
          {eMsg("No keys named " <<Opt.KeyName <<" found to install.");
           return 8;
          }
      }

// Now rewrite the file
//
   Opt.kTab->setPath(Opt.KeyFile);
   if ((retc = Opt.kTab->Rewrite(Opt.Keep, numKeys, numTot, numExp)))
      {eMsg("Unable to install keytab '" <<Opt.KeyFile <<"'; " <<strerror(retc));
       retc = 8;
      } else {
       eMsg(numKeys <<(numKeys == 1 ? " key":" keys") <<" out of "
            <<numTot <<" installed (" <<numExp <<" expired).");
      }

// All done
//
   return retc;
}

/******************************************************************************/
/*                  X r d S e c s s s A d m i n _ i s K e y                   */
/******************************************************************************/
  
int  XrdSecsssAdmin_isKey(XrdsecsssAdmin_Opts &Opt,
                          XrdSecsssKT::ktEnt *ktP)
{
   if (Opt.KeyName && strcmp(ktP->Data.Name, Opt.KeyName)) return 0;
   if (Opt.KeyUser && strcmp(ktP->Data.User, Opt.KeyUser)) return 0;
   if (Opt.KeyGrup && strcmp(ktP->Data.Grup, Opt.KeyGrup)) return 0;
   return 1;
}

/******************************************************************************/
/*                   X r d S e c s s s A d m i n _ H e r e                    */
/******************************************************************************/
  
int XrdSecsssAdmin_Here(char sType, XrdSecsssKT::ktEnt *ktX,
                                    XrdSecsssKT::ktEnt *ktS)
{
   int n;
   char *sf1, *sf2;

   switch(sType)
         {case 'c': return ktX->Data.Crt < ktS->Data.Crt;
          case 'g': sf1 = ktX->Data.Grup; sf2 = ktS->Data.Grup; break;
          case 'k': sf1 = ktX->Data.Name; sf2 = ktS->Data.Name; break;
          case 'n': return (ktX->Data.ID & 0x7fffffff) < (ktS->Data.ID & 0x7fffffff);
          case 'u': sf1 = ktX->Data.User; sf2 = ktS->Data.User; break;
          case 'x': return ktX->Data.Exp < ktS->Data.Exp;
          default:  return 0;
         }

   if ((n = strcmp(sf1, sf2))) return n < 0;
   return (ktX->Data.ID & 0x7fffffff) < (ktS->Data.ID & 0x7fffffff);
}

/******************************************************************************/
/*                 X r d S e c s s s A d m i n _ l s t K e y                  */
/******************************************************************************/
  
int  XrdSecsssAdmin_lstKey(XrdsecsssAdmin_Opts &Opt)
{
   static const char Hdr1[] =
   "     Number Len Date/Time Created Expires  Keyname User & Group\n";
//  12345678901 123 mm/dd/yy hh:mm:ss mm/dd/yy
   static const char Hdr2[] =
   "     ------ --- --------- ------- -------- -------\n";

   extern int XrdSecsssAdmin_isKey(XrdsecsssAdmin_Opts &Opt,
                                   XrdSecsssKT::ktEnt *ktP);
   XrdOucErrInfo eInfo;
   XrdSecsssKT::ktEnt *ktP, *ktSort = 0, *ktS, *ktSP, *ktX;
   char crfmt[] = "%D %T", exfmt[] = "%D";
   char buff[128], crbuff[64], exbuff[16];
   int retc, pHdr = 1;

// Allocate the initial keytab
//
   Opt.kTab = new XrdSecsssKT(&eInfo, Opt.KeyFile, XrdSecsssKT::isAdmin);
   if ((retc = eInfo.getErrInfo()))
      {if (retc == ENOENT)
          {eMsg("Keyfile '" <<Opt.KeyFile <<"' does not exist.");}
       return 4;
      }

// Obtain the keytab list
//
   if ((ktP = Opt.kTab->keyList()))
      {ktSort = ktP; ktP = ktP->Next; ktSort->Next = 0;}

// Sort the list
//
   while(ktP)
        {ktS = ktSort; ktSP = 0; ktX = ktP; ktP = ktP->Next; ktX->Next = 0;
         while(ktS)
              {if (XrdSecsssAdmin_Here(Opt.Sort, ktX, ktS))
                  {if (ktSP) {ktX->Next = ktS; ktSP->Next = ktX;}
                      else   {ktX->Next = ktSort; ktSort  = ktX;}
                   break;
                  }
               ktSP = ktS; ktS = ktS->Next;
              }
         if (!ktS) ktSP->Next = ktX;
        }

// List the keys
//
   ktP = ktSort;
   while(ktP)
        {if (XrdSecsssAdmin_isKey(Opt, ktP))
            {if (pHdr) {cout <<Hdr1 <<Hdr2; pHdr = 0;}
             sprintf(buff, "%11lld %3d ", (ktP->Data.ID & 0x7fffffff), ktP->Data.Len);
             strftime(crbuff, sizeof(crbuff), crfmt, localtime(&ktP->Data.Crt));
             if (!ktP->Data.Exp) strcpy(exbuff, "--------");
                else strftime(exbuff,sizeof(exbuff),exfmt,localtime(&ktP->Data.Exp));
             cout <<buff <<crbuff <<' ' <<exbuff <<' ' <<ktP->Data.Name <<' '
                  <<ktP->Data.User <<' ' <<ktP->Data.Grup <<endl;
            }
         ktP = ktP->Next;
        }

// Check if we printed anything
//
   if (pHdr)
      {if (Opt.KeyName) eMsg(Opt.KeyName <<" key not found in " <<Opt.KeyFile);
          else eMsg("No keys found in " <<Opt.KeyFile);
      }
   return 0;
}


