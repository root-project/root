/******************************************************************************/
/*                                                                            */
/*                        X r d S e c s s s K T . c c                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

const char *XrdSecsssKTCVSID = "$Id$";

#include <fcntl.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdSecsss/XrdSecsssKT.hh"

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysHeaders.hh"
  
/******************************************************************************/
/*                    S t a t i c   D e f i n i t i o n s                     */
/******************************************************************************/

int XrdSecsssKT::randFD = -1;
  
/******************************************************************************/
/*                       X r d S e c s s s K T R e f r                        */
/******************************************************************************/
  
void *XrdSecsssKTRefresh(void *Data)
{
   XrdSecsssKT *theKT = (XrdSecsssKT *)Data;
   struct timespec naptime = {theKT->RefrTime(), 0};

// Loop and check if keytab has changed
//
   while(1) {nanosleep(&naptime, 0); theKT->Refresh();}

   return (void *)0;
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdSecsssKT::XrdSecsssKT(XrdOucErrInfo *eInfo, const char *kPath,
                         xMode oMode, int refrInt)
{
   static const char *eText = "Unable to start keytab refresh thread";
   const char *devRand = "/dev/urandom";
   struct stat sbuf;
   pthread_t pid;
   int retc;

// Do some common initialization
//
   ktPath = (kPath ? strdup(kPath) : 0);
   ktList = 0; kthiID = 0; ktMode = oMode; ktRefT = (time_t)refrInt;
   if (eInfo) eInfo->setErrCode(0);

// Prepare /dev/random if we have it
//
   if (stat(devRand, &sbuf)) devRand = "/dev/random";
   if ((randFD = open(devRand, O_RDONLY)) < 0
   && oMode != isClient && errno != ENOENT)
      eMsg("sssKT",errno,"Unable to generate random key"," opening ",devRand);

// First get the stat information for the file
//
   if (!kPath)
      {if (oMode != isAdmin)
          {eMsg("sssKT", -1, "Keytable path not specified.");
           if (eInfo) eInfo->setErrInfo(EINVAL, "Keytable path missing.");
           return;
          }
       sbuf.st_mtime = 0; sbuf.st_mode = S_IRWXU;
      } else if (stat(kPath, &sbuf))
                {if (eInfo) eInfo->setErrInfo(errno, "Keytable not found");
                 if (errno != ENOENT || oMode != isAdmin)
                    eMsg("sssKT",errno,"Unable process keytable ",kPath);
                 return;
                }

// Now read in the whole key table and start possible refresh thread
//
   if ((ktList = getKeyTab(eInfo, sbuf.st_mtime, sbuf.st_mode))
   && (oMode != isAdmin) && (!eInfo || eInfo->getErrInfo() == 0))
      {if ((retc = XrdSysThread::Run(&pid,XrdSecsssKTRefresh,(void *)this)))
          {eMsg("sssKT", errno, eText); eInfo->setErrInfo(-1, eText);}
      }
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdSecsssKT::~XrdSecsssKT()
{
   ktEnt *ktP;

   myMutex.Lock();
   if (ktPath) {free(ktPath); ktPath = 0;}

   while((ktP = ktList)) {ktList = ktList->Next; delete ktP;}
   myMutex.UnLock();
}
  
/******************************************************************************/
/*                               a d d K e y                                  */
/******************************************************************************/
  
void XrdSecsssKT::addKey(ktEnt &ktNew)
{
   ktEnt *ktPP = 0, *ktP;

// Generate a key for this entry
//
   genKey(ktNew.Data.Val, ktNew.Data.Len);
   ktNew.Data.Crt = time(0);
   ktNew.Data.ID  = static_cast<long long>(ktNew.Data.Crt & 0x7fffffff) << 32L
                  | static_cast<long long>(++kthiID);

// Locate place to insert this key
//
   ktP = ktList;
   while(ktP && !isKey(*ktP, &ktNew, 0)) {ktPP = ktP; ktP = ktP->Next;}

// Now chain in the entry
//
   if (ktPP) ktPP->Next = &ktNew;
      else   ktList     = &ktNew;
   ktNew.Next = ktP;
}

/******************************************************************************/
/*                                d e l K e y                                 */
/******************************************************************************/
  
int XrdSecsssKT::delKey(ktEnt &ktDel)
{
   ktEnt *ktN, *ktPP = 0, *ktP = ktList;
   int nDel = 0;

// Remove all matching keys
//
   while(ktP)
        {if (isKey(ktDel, ktP))
            {if (ktPP) ktPP->Next = ktP->Next;
                else   ktList     = ktP->Next;
             ktN = ktP; ktP = ktP->Next; delete ktN; nDel++;
            } else {ktPP = ktP; ktP = ktP->Next;}
        }

   return nDel;
}

/******************************************************************************/
/*                                g e t K e y                                 */
/******************************************************************************/
  
int XrdSecsssKT::getKey(ktEnt &theEnt)
{
   ktEnt *ktP, *ktN;

// Lock the keytab to prevent modification
//
   myMutex.Lock();
   ktP = ktList;

// Find first key by key name (used normally by clients) or by keyID
//
   if (!*theEnt.Data.Name)
      {if (theEnt.Data.ID >= 0) 
          while(ktP && ktP->Data.ID != theEnt.Data.ID)  ktP = ktP->Next;
      }
      else {while(ktP && strcmp(ktP->Data.Name,theEnt.Data.Name)) ktP=ktP->Next;
            while(ktP && ktP->Data.Exp <= time(0))
                 {if (!(ktN=ktP->Next) 
                  ||  strcmp(ktN->Data.Name,theEnt.Data.Name)) break;
                  ktP = ktN;
                 }
           }

// If we found a match, export it
//
   if (ktP) theEnt = *ktP;
   myMutex.UnLock();

// Indicate if key expired
//
   if (!ktP) return ENOENT;
   return (theEnt.Data.Exp && theEnt.Data.Exp <= time(0) ? -1 : 0);
}

/******************************************************************************/
/*                                 g e n F N                                  */
/******************************************************************************/
  
char *XrdSecsssKT::genFN()
{
   static char fnbuff[1040];
   const char *pfx;

// Get the path prefix
//
   if (!(pfx = getenv("HOME")) || !*pfx) pfx = "";

// Format the name
//
   snprintf(fnbuff, sizeof(fnbuff), "%s/.xrd/sss.keytab", pfx);
   return fnbuff;
}
  
/******************************************************************************/
/*                                g e n K e y                                 */
/******************************************************************************/
  
void XrdSecsssKT::genKey(char *kBP, int kLen)
{
   struct timeval tval;
   int kTemp;

// See if we can directly service the key. Make sure that we get some entropy
// because some /dev/random devices start out really cold.
//
   if (randFD >= 0) 
      {char *buffP = kBP;
       int i, Got, Want = kLen, zcnt = 0, maxZ = kLen*25/100;
       while(Want)
       do { {do {Got = read(randFD, buffP, Want);}
                while(Got < 0 && errno == EINTR);
             if (Got > 0) {buffP += Got; Want -= Got;}
            }
          } while(Got > 0 && Want);
       if (!Want)
          {for (i = 0; i < kLen; i++) if (!kBP[i]) zcnt++;
           if (zcnt <= maxZ) return;
          }
      }

// Generate a seed
//
   gettimeofday(&tval, 0);
   if (tval.tv_usec == 0) tval.tv_usec = tval.tv_sec;
   tval.tv_usec = tval.tv_usec ^ getpid();
   srand48(static_cast<long>(tval.tv_usec));

// Now generate the key (we ignore he fact that longs may be 4 or 8 bytes)
//
   while(kLen > 0)
        {kTemp = mrand48();
         memcpy(kBP, &kTemp, (4 > kLen ? kLen : 4));
         kBP += 4; kLen -= 4;
        }
}

/******************************************************************************/
/*                               R e f r e s h                                */
/******************************************************************************/
  
void XrdSecsssKT::Refresh()
{
   XrdOucErrInfo eInfo;
   ktEnt *ktNew, *ktOld, *ktNext;
   struct stat sbuf;
   int retc = 0;

// Get change time of keytable and if changed, update it
//
   if (stat(ktPath, &sbuf) == 0)
      {if (sbuf.st_mtime == ktMtime) return;
       if ((ktNew = getKeyTab(&eInfo, sbuf.st_mtime, sbuf.st_mode))
       && eInfo.getErrInfo() == 0)
          {myMutex.Lock(); ktOld = ktList; ktList = ktNew; myMutex.UnLock();
          } else ktOld = ktNew;
       while(ktOld) {ktNext = ktOld->Next; delete ktOld; ktOld = ktNext;}
       if ((retc == eInfo.getErrInfo()) == 0) return;
      } else retc = errno;

// Refresh failed
//
   eMsg("Refresh",retc,"Unable to refresh keytable",ktPath);
}

/******************************************************************************/
/*                               R e w r i t e                                */
/******************************************************************************/
  
int XrdSecsssKT::Rewrite(int Keep, int &numKeys, int &numTot, int &numExp)
{
   char tmpFN[1024], buff[2048], kbuff[4096], *Slash;
   int ktFD, numID = 0, n, retc = 0;
   ktEnt ktCurr, *ktP, *ktN;
   mode_t theMode = fileMode(ktPath);

// Invoke mkpath in case the path is missing
//
   strcpy(tmpFN, ktPath);
   if ((Slash = rindex(tmpFN, '/'))) *Slash = '\0';
   retc = XrdOucUtils::makePath(tmpFN,S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH);
   if (retc) return (retc < 0 ? -retc : retc);
   if (Slash) *Slash = '/';

// Construct temporary filename
//
   sprintf(buff, ".%d", static_cast<int>(getpid()));
   strcat(tmpFN, buff);

// Open the file for output
//
   if ((ktFD = open(tmpFN, O_WRONLY|O_CREAT|O_TRUNC, theMode)) < 0)
      return errno;

// Write all of the keytable
//
   ktCurr.Data.Name[0] = ktCurr.Data.User[0] = ktCurr.Data.Grup[0] = 3;
   ktN = ktList; numKeys = numTot = numExp = 0;
   while((ktP = ktN))
        {ktN = ktN->Next; numTot++;
         if (ktP->Data.Name[0] == '\0') continue;
         if (ktP->Data.Exp && ktP->Data.Exp <= time(0)) {numExp++; continue;}
         if (!isKey(ktCurr, ktP, 0)) {ktCurr.NUG(ktP); numID = 0;}
            else if (Keep && numID >= Keep) continue;
         n = sprintf(buff, "%s0 u:%s g:%s n:%s N:%lld c:%ld e:%ld k:",
                    (numKeys ? "\n" : ""),
                     ktP->Data.User,ktP->Data.Grup,ktP->Data.Name,ktP->Data.ID,
                     ktP->Data.Crt, ktP->Data.Exp);
         numID++; numKeys++; keyB2X(ktP, kbuff);
         if (write(ktFD, buff, n) < 0
         ||  write(ktFD, kbuff, ktP->Data.Len*2) < 0) break;
        }

// Check for errors
//
   if (ktP) retc = errno;
      else if (!numKeys) retc = ENODATA;

// Atomically trounce the original file if we can
//
   close(ktFD);
   if (!retc && rename(tmpFN, ktPath) < 0) retc = errno;

// All done
//
   unlink(tmpFN); 
   return retc;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                  e M s g                                   */
/******************************************************************************/

int XrdSecsssKT::eMsg(const char *epname, int rc,
                      const char *txt1, const char *txt2,
                      const char *txt3, const char *txt4)
{
              cerr <<"Secsss (" << epname <<"): ";
              cerr <<txt1;
   if (txt2)  cerr <<txt2;
   if (txt3)  cerr <<txt3;
   if (txt4)  cerr <<txt4;
   if (rc>0)  cerr <<"; " <<strerror(rc);
              cerr <<endl;

   return (rc ? (rc < 0 ? rc : -rc) : -1);
}

/******************************************************************************/
/*                             g e t K e y T a b                              */
/******************************************************************************/
  
XrdSecsssKT::ktEnt* XrdSecsssKT::getKeyTab(XrdOucErrInfo *eInfo,
                                           time_t Mtime, mode_t Amode)
{
   static const int altMode = S_IRWXG | S_IRWXO;
   XrdOucStream myKT;
   int ktFD, retc, tmpID, recno = 0, NoGo = 0;
   const char *What = 0, *ktFN;
   char *lp, *tp, rbuff[64];
   ktEnt *ktP, *ktPP, *ktNew, *ktBase = 0;

// Verify that the keytable is only readable by us
//
   ktMtime = Mtime;
   if ((Amode & altMode) & ~fileMode(ktPath))
      {if (eInfo) eInfo->setErrInfo(EACCES, "Keytab file is not secure!");
       eMsg("getKeyTab",-1,"Unable to process ",ktPath,"; file is not secure!");
       return 0;
      }

// Open the file
//
   if (ktPath)
      {if ((ktFD = open(ktPath, O_RDONLY)) < 0)
          {if (eInfo) eInfo->setErrInfo(errno, "Unable to open keytab file.");
           eMsg("getKeyTab", errno, "Unable to open ", ktPath);
           return 0;
          } else ktFN = ktPath;
      } else {ktFD = dup(STDIN_FILENO); ktFN = "stdin";}

// Attach the fd to the stream
//
   myKT.Attach(ktFD);

// Now start reading the keytable which always has the form:
//
// <format> <whatever data based on format>
//
do{while((lp = myKT.GetLine()))
        {recno++; What = 0;
         if (!*lp) continue;
         if (!(tp = myKT.GetToken()) || strcmp("0", tp))
            {What = "keytable format missing or unsupported";      break;}
         if (!(ktNew = ktDecode0(myKT, eInfo)))
            {What = (eInfo ? eInfo->getErrText(): "invalid data"); break;}
         if (ktMode!=isAdmin && ktNew->Data.Exp && ktNew->Data.Exp <= time(0))
            {delete ktNew; continue;}
         tmpID = static_cast<int>(ktNew->Data.ID & 0x7fffffff);
         if (tmpID > kthiID) kthiID = tmpID;
         
         ktP = ktBase; ktPP = 0;
         while(ktP && !isKey(*ktP, ktNew, 0)) {ktPP=ktP; ktP=ktP->Next;}
         if (!ktP) {ktNew->Next = ktBase; ktBase = ktNew;}
            else {if (ktMode == isClient)
                     {if ((ktNew->Data.Exp == 0 && ktP->Data.Exp != 0)
                      ||  (ktP->Data.Exp!=0 && ktP->Data.Exp < ktNew->Data.Exp))
                          ktP->Set(*ktNew);
                      delete ktNew;
                     } else {
                      while(ktNew->Data.Crt < ktP->Data.Crt)
                           {ktPP = ktP; ktP = ktP->Next;
                            if (!ktP || !isKey(*ktP, ktNew, 0)) break;
                           }
                      if (ktPP) {ktPP->Next = ktNew; ktNew->Next = ktP;}
                         else   {ktNew->Next= ktBase; ktBase = ktNew;}
                     }
                 }
        }
   if (What)
      {sprintf(rbuff, "; line %d in ", recno);
       NoGo = eMsg("getKeyTab", -1, What, rbuff, ktFN);
      }
  } while(lp);

// Check for problems
//
        if (NoGo) {if (eInfo) eInfo->setErrInfo(EINVAL,"Invalid keytab file.");}
   else if ((retc = myKT.LastError()))
           {if (eInfo) eInfo->setErrInfo(retc,"Unable to read keytab file.");
            NoGo = eMsg("getKeyTab", retc, "Unable to read keytab ",ktFN);
           }
   else if (!ktBase)
           {if (eInfo) eInfo->setErrInfo(ESRCH,"Keytable is empty.");
            NoGo = eMsg("getKeyTab",-1,"No keys found in ",ktFN);
           }


// Check if an error should be returned
//
   if (!NoGo) eInfo->setErrCode(0);

// All done
//
   myKT.Close();
   return ktBase;
}

/******************************************************************************/
/*                               g r p F i l e                                */
/******************************************************************************/
  
mode_t XrdSecsssKT::fileMode(const char *Path)
{
   int n;

   return (!Path || (n = strlen(Path)) < 5 || strcmp(".grp", &Path[n-4])
        ? S_IRUSR|S_IWUSR : S_IRUSR|S_IWUSR|S_IRGRP);
}

/******************************************************************************/
/*                                 i s K e y                                  */
/******************************************************************************/

int  XrdSecsssKT::isKey(ktEnt &ktRef, ktEnt *ktP, int Full)
{
   if (*ktRef.Data.Name && strcmp(ktP->Data.Name, ktRef.Data.Name)) return 0;
   if (*ktRef.Data.User && strcmp(ktP->Data.User, ktRef.Data.User)) return 0;
   if (*ktRef.Data.Grup && strcmp(ktP->Data.Grup, ktRef.Data.Grup)) return 0;
   if (Full && ktRef.Data.ID > 0
   && (ktP->Data.ID & 0x7fffffff) != ktRef.Data.ID) return 0;
   return 1;
}
  
/******************************************************************************/
/*                                k e y B 2 X                                 */
/******************************************************************************/

void XrdSecsssKT::keyB2X(ktEnt *theKT, char *buff)
{
   static const char xTab[] = "0123456789abcdef";
   int  kLen = theKT->Data.Len;
   char *kP  = theKT->Data.Val, Val;

// Convert
//
   while(kLen--)
        {Val = *kP++;
         *buff++ = xTab[(Val>>4) & 0x0f];
         *buff++ = xTab[ Val     & 0x0f];
        }
   *buff = '\0';
}
  
/******************************************************************************/
/*                                k e y X 2 B                                 */
/******************************************************************************/
  
void XrdSecsssKT::keyX2B(ktEnt *theKT, char *xKey)
{
//                              0   1   2   3   4   5   6   7
   static const char xtab[] = {10, 10, 11, 12, 13, 14, 15, 15};
   int n = strlen(xKey);
   char *kp, kByte;

// Make sure we don't overflow
//
   n = (n%2 ? (n+1)/2 : n/2);
   if (n > ktEnt::maxKLen) n = ktEnt::maxKLen;
   kp = theKT->Data.Val;
   theKT->Data.Val[n-1] = 0;

// Now convert (we need this to be just consistent not necessarily correct)
//
   while(*xKey)
        {if (*xKey <= '9') kByte  = (*xKey & 0x0f) << 4;
            else kByte = xtab[*xKey & 0x07] << 4;
         xKey++;
         if (*xKey <= '9') kByte |= (*xKey & 0x0f);
            else kByte |= xtab[*xKey & 0x07];
         *kp++ = kByte; xKey++;
        }

// Return data via the structure
//
   theKT->Data.Len = n;
}

/******************************************************************************/
/*                             k t D e c o d e 0                              */
/******************************************************************************/
  
XrdSecsssKT::ktEnt *XrdSecsssKT::ktDecode0(XrdOucStream  &kTab,
                                           XrdOucErrInfo *eInfo)
{
   static const short haveCRT = 0x0001;
   static const short haveEXP = 0x0002;
   static const short isTIMET = 0x0003;
   static const short haveGRP = 0x0004;
   static const short haveKEY = 0x0008;
   static const short haveNAM = 0x0010;
   static const short haveNUM = 0x0020;
   static const short haveUSR = 0x0040;

   static struct 
          {const char *Name; size_t Offset; int Ctl; short What; char Tag;}
          ktDesc[] = {
   {"crtdt",   offsetof(ktEnt::ktData,Crt),  0,                haveCRT, 'c'},
   {"expdt",   offsetof(ktEnt::ktData,Exp),  0,                haveEXP, 'e'},
   {"group",   offsetof(ktEnt::ktData,Grup), ktEnt::GrupSZ,    haveGRP, 'g'},
   {"keyval",  offsetof(ktEnt::ktData,Val),  ktEnt::maxKLen*2, haveKEY, 'k'},
   {"keyname", offsetof(ktEnt::ktData,Name), ktEnt::NameSZ,    haveNAM, 'n'},
   {"keynum",  offsetof(ktEnt::ktData,ID),   0,                haveNUM, 'N'},
   {"user",    offsetof(ktEnt::ktData,User), ktEnt::UserSZ,    haveUSR, 'u'}
   };
   static const int ktDnum = sizeof(ktDesc)/sizeof(ktDesc[0]);

   ktEnt *ktNew = new ktEnt;
   const char *Prob = 0, *What = "Whatever";
   char Tag, *Dest, *ep, *tp;
   long long nVal;
   short Have = 0;
   int i = 0;

// Decode the record using the tags described in the above table
//
while((tp = kTab.GetToken()) && !Prob)
     {Tag = *tp++;
      if (*tp++ == ':')
         for (i = 0; i < ktDnum; i++)
             if (ktDesc[i].Tag == Tag)
                {Dest = (char *)&(ktNew->Data) + ktDesc[i].Offset;
                 Have |= ktDesc[i].What; What = ktDesc[i].Name;
                 if (ktDesc[i].Ctl)
                    {if ((int)strlen(tp) > ktDesc[i].Ctl) Prob=" is too long";
                        else if (Tag == 'k') keyX2B(ktNew, tp);
                                else strcpy(Dest, tp);
                    } else {
                     nVal = strtoll(tp, &ep, 10);
                     if (ep && *ep) Prob = " has invalid value";
                        else if (ktDesc[i].What & isTIMET)
                                 *(time_t *)Dest = static_cast<time_t>(nVal);
                                else *(long long *)Dest = nVal;
                    }
                }
     }

// If no problem, make sure we have the essential elements
//
   if (!Prob)
      {if (!(Have & haveGRP)) strcpy(ktNew->Data.Grup, "nogroup");
       if (!(Have & haveNAM)) strcpy(ktNew->Data.Name, "nowhere");
       if (!(Have & haveUSR)) strcpy(ktNew->Data.User, "nobody");
            if (!(Have & haveKEY)) {What = "keyval"; Prob = " not found";}
       else if (!(Have & haveNUM)) {What = "keynum"; Prob = " not found";}
      }

// Check if we have a problem
//
   if (Prob)
      {const char *eVec[] = {What, Prob};
       if (eInfo) eInfo->setErrInfo(-1, eVec, 2);
       delete ktNew;
       return 0;
      }

// Set special value options
//
   if (!strcmp(ktNew->Data.Grup, "anygroup"))       
      ktNew->Data.Opts|=ktEnt::anyGRP;
      else if (!strcmp(ktNew->Data.Grup, "usrgroup"))
              ktNew->Data.Opts|=ktEnt::usrGRP;
   if (!strcmp(ktNew->Data.User, "anybody"))
      ktNew->Data.Opts|=ktEnt::anyUSR;

// All done
//
   return ktNew;
}
