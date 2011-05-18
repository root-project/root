/******************************************************************************/
/*                                                                            */
/*                       X r d A c c C o n f i g . c c                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

const char *XrdAccConfigCVSID = "$Id$";

/*
   The routines in this file handle authorization system initialization.

   These routines are thread-safe if compiled with:
   AIX: -D_THREAD_SAFE
   SUN: -D_REENTRANT
*/
  
#include <unistd.h>
#include <ctype.h>
#include <fcntl.h>
#include <strings.h>
#include <stdio.h>
#include <time.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdOuc/XrdOucLock.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdAcc/XrdAccAccess.hh"
#include "XrdAcc/XrdAccAudit.hh"
#include "XrdAcc/XrdAccConfig.hh"
#include "XrdAcc/XrdAccGroups.hh"
#include "XrdAcc/XrdAccCapability.hh"

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/
  
// The following is the single configuration object. Other objects needing
// access to this object should simply declare an extern to it.
//
XrdAccConfig XrdAccConfiguration;

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define TS_Xeq(x,m)   if (!strcmp(x,var)) return m(Config,Eroute);

#define TS_Str(x,m)   if (!strcmp(x,var)) {free(m); m = strdup(val); return 0;}

#define TS_Chr(x,m)   if (!strcmp(x,var)) {m = val[0]; return 0;}

#define TS_Bit(x,m,v) if (!strcmp(x,var)) {m |= v; return 0;}

#define ACC_PGO 0x0001

/******************************************************************************/
/*                    E x t e r n a l   F u n c t i o n s                     */
/******************************************************************************/
/******************************************************************************/
/*                  o o a c c _ C o n f i g _ R e f r e s h                   */
/******************************************************************************/

void *XrdAccConfig_Refresh( void *start_data )
{
   XrdSysError *Eroute = (XrdSysError *)start_data;

// Get the number of seconds between refreshes
//
   struct timespec naptime = {(time_t)XrdAccConfiguration.AuthRT, 0};

// Now loop until the bitter end
//
   while(1)
        {nanosleep(&naptime, 0); XrdAccConfiguration.ConfigDB(1, *Eroute);}
   return (void *)0;
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdAccConfig::XrdAccConfig()
{

// Initialize path value and databse pointer to nil
//
   dbpath        = strdup("/opt/xrd/etc/Authfile");
   Database      = 0;
   Authorization = 0;

// Establish other defaults
//
   ConfigDefaults();
}

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdAccConfig::Configure(XrdSysError &Eroute, const char *cfn) {
/*
  Function: Establish default values using a configuration file.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   char *var;
   int  retc, NoGo = 0, Cold = (Database == 0);
   pthread_t reftid;

// Print warm-up message
//
   Eroute.Say("++++++ Authorization system initialization started.");

// Process the configuration file and authorization database
//
   if (!(Authorization = new XrdAccAccess(&Eroute))
   ||   (NoGo = ConfigFile(Eroute, cfn))
   ||   (NoGo = ConfigDB(0, Eroute)))
       {if (Authorization) {delete Authorization, Authorization = 0;}
        NoGo = 1;
       }

// Start a refresh thread unless this was a refresh thread call
//
   if (Cold && !NoGo)
      {if ((retc=XrdSysThread::Run(&reftid,XrdAccConfig_Refresh,(void *)&Eroute)))
          Eroute.Emsg("ConfigDB",retc,"start refresh thread.");
      }

// All done
//
   var = (NoGo > 0 ? (char *)"failed." : (char *)"completed.");
   Eroute.Say("------ Authorization system initialization ", var);
   return (NoGo > 0);
}
  
/******************************************************************************/
/*                              C o n f i g D B                               */
/******************************************************************************/
  
int XrdAccConfig::ConfigDB(int Warm, XrdSysError &Eroute)
{
/*
  Function: Establish default values using a configuration file.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   char buff[128];
   int  retc, anum = 0, NoGo = 0;
   struct XrdAccAccess_Tables tabs;
   XrdOucLock cdb_Lock(&Config_Context);

// Indicate type of start we are doing
//
   if (!Database) NoGo = !(Database = XrdAccAuthDBObject(&Eroute));
      else if (Warm && !Database->Changed(dbpath)) return 0;

// Try to open the authorization database
//
   if (!Database || !Database->Open(Eroute, dbpath)) return 1;

// Allocate new hash tables
//
   if (!(tabs.G_Hash = new XrdOucHash<XrdAccCapability>()) ||
       !(tabs.H_Hash = new XrdOucHash<XrdAccCapability>()) ||
       !(tabs.N_Hash = new XrdOucHash<XrdAccCapability>()) ||
       !(tabs.T_Hash = new XrdOucHash<XrdAccCapability>()) ||
       !(tabs.U_Hash = new XrdOucHash<XrdAccCapability>()) )
      {Eroute.Emsg("ConfigDB","Insufficient storage for id tables.");
       Database->Close(); return 1;
      }

// Now start processing records until eof.
//
   while((retc = ConfigDBrec(Eroute, tabs))) {NoGo |= retc < 0; anum++;}
   snprintf(buff, sizeof(buff), "%d auth entries processed in ", anum);
   Eroute.Say("Config ", buff, dbpath);

// All done, close the database and return if we failed
//
   if (!Database->Close() || NoGo) return 1;

// Set the access control tables
//
   if (!tabs.G_Hash->Num()) {delete tabs.G_Hash; tabs.G_Hash=0;}
   if (!tabs.H_Hash->Num()) {delete tabs.H_Hash; tabs.H_Hash=0;}
   if (!tabs.N_Hash->Num()) {delete tabs.N_Hash; tabs.N_Hash=0;}
   if (!tabs.T_Hash->Num()) {delete tabs.T_Hash; tabs.T_Hash=0;}
   if (!tabs.U_Hash->Num()) {delete tabs.U_Hash; tabs.U_Hash=0;}
   Authorization->SwapTabs(tabs);

// All done
//
   return NoGo;
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*        C o n f i g   F i l e   P r o c e s s i n g   M e t h o d s         */
/******************************************************************************/
  
int XrdAccConfig::ConfigFile(XrdSysError &Eroute, const char *ConfigFN) {
/*
  Function: Establish default values using a configuration file.

  Input:    None.

  Output:   1 - Processing failed.
            0 - Processing completed successfully.
           -1 = Security is to be disabled by request.
*/
   char *var;
   int  cfgFD, retc, NoGo = 0, recs = 0;
   XrdOucEnv myEnv;
   XrdOucStream Config(&Eroute, getenv("XRDINSTANCE"), &myEnv, "=====> ");

// If there is no config file, complain
//
   if( !ConfigFN || !*ConfigFN)
     {Eroute.Emsg("Config", "Authorization configuration file not specified.");
      return 1;
     } 

// Check if security is to be disabled
//
   if (!strcmp(ConfigFN, "none"))
      {Eroute.Emsg("Config", "Authorization system deactivated.");
       return -1;
      }

// Try to open the configuration file.
//
   if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
      {Eroute.Emsg("Config", errno, "open config file", ConfigFN);
       return 1;
      }
   Eroute.Emsg("Config","Authorization system using configuration in",ConfigFN);

// Now start reading records until eof.
//
   ConfigDefaults(); Config.Attach(cfgFD); Config.Tabs(0);
   while((var = Config.GetMyFirstWord()))
        {if (!strncmp(var, "acc.", 2))
            {recs++;
             if (ConfigXeq(var+4, Config, Eroute)) {Config.Echo(); NoGo = 1;}
            }
        }

// Now check if any errors occured during file i/o
//
   if ((retc = Config.LastError()))
      NoGo = Eroute.Emsg("Config",-retc,"read config file",ConfigFN);
      else {char buff[128];
            snprintf(buff, sizeof(buff), 
                     "%d authorization directives processed in ", recs);
            Eroute.Say("Config ", buff, ConfigFN);
           }
   Config.Close();

// Set external options, as needed
//
   if (options & ACC_PGO) GroupMaster.SetOptions(Primary_Only);

// All done
//
   return NoGo;
}

/******************************************************************************/
/*                        C o n f i g D e f a u l t s                         */
/******************************************************************************/

void XrdAccConfig::ConfigDefaults()
{
   AuthRT   = 60*60*12;
   options  = 0;
}
  
/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/
  
int XrdAccConfig::ConfigXeq(char *var, XrdOucStream &Config, XrdSysError &Eroute)
{

// Fan out based on the variable
//
   TS_Xeq("audit",         xaud);
   TS_Xeq("authdb",        xdbp);
   TS_Xeq("authrefresh",   xart);
   TS_Xeq("gidlifetime",   xglt);
   TS_Xeq("gidretran",     xgrt);
   TS_Xeq("nisdomain",     xnis);
   TS_Bit("pgo",           options, ACC_PGO);

// No match found, complain.
//
   Eroute.Emsg("Config", "unknown directive", var);
   Config.Echo();
   return 1;
}
  
/******************************************************************************/
/*                                  x a u d                                   */
/******************************************************************************/

/* Function: xaud

   Purpose:  To parse the directive: audit <options>

             options:

             deny     audit access denials.
             grant    audit access grants.
             none     audit is disabled.

   Output: 0 upon success or !0 upon failure.
*/

int XrdAccConfig::xaud(XrdOucStream &Config, XrdSysError &Eroute)
{
    static struct auditopts {const char *opname; int opval;} audopts[] =
       {
        {"deny",     (int)audit_deny},
        {"grant",    (int)audit_grant}
       };
    int i, audval = 0, numopts = sizeof(audopts)/sizeof(struct auditopts);
    char *val;

    val = Config.GetWord();
    if (!val || !val[0])
       {Eroute.Emsg("Config", "audit option not specified"); return 1;}
    while (val && val[0])
          {if (!strcmp(val, "none")) audval = (int)audit_none;
              else for (i = 0; i < numopts; i++)
                       {if (!strcmp(val, audopts[i].opname))
                           {audval |= audopts[i].opval; break;}
                        if (i >= numopts)
                           {Eroute.Emsg("Config","invalid audit option -",val);
                            return 1;
                           }
                       }
          val = Config.GetWord();
         }
    Authorization->Auditor->setAudit((XrdAccAudit_Options)audval);
    return 0;
}

/******************************************************************************/
/*                                  x a r t                                   */
/******************************************************************************/

/* Function: xart

   Purpose:  To parse the directive: authrefresh <seconds>

             <seconds> minimum number of seconds between aythdb refreshes.

   Output: 0 upon success or !0 upon failure.
*/

int XrdAccConfig::xart(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int reft;

      val = Config.GetWord();
      if (!val || !val[0])
         {Eroute.Emsg("Config","authrefresh value not specified");return 1;}
      if (XrdOuca2x::a2tm(Eroute,"authrefresh value",val,&reft,60))
         return 1;
      AuthRT = reft;
      return 0;
}

/******************************************************************************/
/*                                  x d b p                                   */
/******************************************************************************/

/* Function: xdbp

   Purpose:  To parse the directive: authdb <path>

             <path>    is the path to the authorization database.

   Output: 0 upon success or !0 upon failure.
*/

int XrdAccConfig::xdbp(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;

      val = Config.GetWord();
      if (!val || !val[0])
         {Eroute.Emsg("Config","authdb path not specified");return 1;}
      dbpath = strdup(val);
      return 0;
}
  
/******************************************************************************/
/*                                  x g l t                                   */
/******************************************************************************/

/* Function: xglt

   Purpose:  To parse the directive: gidlifetime <seconds>

             <seconds> maximum number of seconds to cache gid information.

   Output: 0 upon success or !0 upon failure.
*/

int XrdAccConfig::xglt(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int reft;

      val = Config.GetWord();
      if (!val || !val[0])
         {Eroute.Emsg("Config","gidlifetime value not specified");return 1;}
      if (XrdOuca2x::a2tm(Eroute,"gidlifetime value",val,&reft,60))
         return 1;
      GroupMaster.SetLifetime(reft);
      return 0;
}

/******************************************************************************/
/*                                  x g r t                                   */
/******************************************************************************/

/* Function: xgrt

   Purpose:  To parse the directive: gidretran <gidlist>

             <gidlist> is a list of blank separated gid's that must be
                       retranslated.

   Output: 0 upon success or !0 upon failure.
*/

int XrdAccConfig::xgrt(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int gid;

    val = Config.GetWord();
    if (!val || !val[0])
       {Eroute.Emsg("Config","gidretran value not specified"); return 1;}

    while (val && val[0])
      {if (XrdOuca2x::a2i(Eroute, "gid", val, &gid, 0)) return 1;
       if (GroupMaster.Retran((gid_t)gid) < 0)
          {Eroute.Emsg("Config", "to many gidretran gid's"); return 1;}
       val = Config.GetWord();
      }
    return 0;
}

/******************************************************************************/
/*                                  x n i s                                   */
/******************************************************************************/

/* Function: xnis

   Purpose:  To parse the directive: nisdomain <domain>

             <domain>  the NIS domain to be used for nis look-ups.

   Output: 0 upon success or !0 upon failure.
*/

int XrdAccConfig::xnis(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;

      val = Config.GetWord();
      if (!val || !val[0])
         {Eroute.Emsg("Config","nisdomain value not specified");return 1;}
      GroupMaster.SetDomain(strdup(val));
      return 0;
}
  
/******************************************************************************/
/*                   D a t a b a s e   P r o c e s s i n g                    */
/******************************************************************************/
/******************************************************************************/
/*                           C o n f i g D B r e c                            */
/******************************************************************************/

int XrdAccConfig::ConfigDBrec(XrdSysError &Eroute,
                            struct XrdAccAccess_Tables &tabs)
{
// The following enum is here for convenience
//
    enum DB_RecType {  Group_ID = 'g',
                        Host_ID = 'h',
                      Netgrp_ID = 'n',
                         Set_ID = 's',
                    Template_ID = 't',
                        User_ID = 'u',
                          No_ID = 0
                    };
    char *authid, rtype, *atype, *path, *privs;
    int alluser = 0, anyuser = 0, domname = 0, NoGo = 0;
    DB_RecType rectype;
    XrdOucHash<XrdAccCapability> *hp;
    XrdAccGroupType gtype = XrdAccNoGroup;
    XrdAccPrivCaps xprivs;
    XrdAccCapability mycap((char *)"", xprivs), *currcap, *lastcap = &mycap;
    XrdAccCapName *ncp;
  
   // Prepare the next record in the database
   //
   if (!(rtype = Database->getRec(&authid))) return 0;
   rectype = (DB_RecType)rtype;

   // Set up to handle the particular record
   //
   switch(rectype)
         {case    Group_ID: hp = tabs.G_Hash; atype = (char *)"group";
                            gtype=XrdAccUnixGroup;
                            break;
          case     Host_ID: hp = tabs.H_Hash; atype = (char *)"host";
                            domname = (authid[0] == '.');
                            break;
          case      Set_ID: hp = 0;           atype = (char *)"set";
                            break;
          case   Netgrp_ID: hp = tabs.N_Hash; atype = (char *)"netgrp";
                            gtype=XrdAccNetGroup;
                            break;
          case Template_ID: hp = tabs.T_Hash; atype = (char *)"template";
                            break;
          case     User_ID: hp = tabs.U_Hash; atype = (char *)"user";
                            alluser = (authid[0] == '*' && !authid[1]);
                            anyuser = (authid[0] == '=' && !authid[1]);
                            break;
                default:    hp = 0;
                            break;
         }

   // Check if we have an invalid or unsupported id-type
   //
   if (!hp) {char badtype[2] = {rtype, '\0'};
             Eroute.Emsg("ConfigXeq", "Invalid id type -", badtype);
             return -1;
            }

   // Check if this id is already defined in the table
   //
   if ((domname && tabs.D_List && tabs.D_List->Find((const char *)authid))
   ||  (alluser && tabs.Z_List) || (anyuser && tabs.X_List) || hp->Find(authid))
      {Eroute.Emsg("ConfigXeq", "duplicate id -", authid);
       return -1;
      }

   // Add this ID to the appropriate group object constants table
   //
   if (gtype) GroupMaster.AddName(gtype, (const char *)authid);

   // Now start getting <path> <priv> pairs until we hit the logical end
   //
   while(1) {NoGo = 0;
             if (!Database->getPP(&path, &privs)) break;
             if (!path) continue;      // Skip pathless entries
             NoGo = 1;
             if (*path != '/')
                {if ((currcap = tabs.T_Hash->Find(path)))
                    currcap = new XrdAccCapability(currcap);
                    else {Eroute.Emsg("ConfigXeq", "Missing template -", path);
                          break;
                         }
                } else {
                  if (!privs)
                     {Eroute.Emsg("ConfigXeq", "Missing privs for path", path);
                      break;
                     }
                  if (!PrivsConvert(privs, xprivs))
                     {Eroute.Emsg("ConfigXeq", "Invalid privs -", privs);
                      break;
                     }
                  currcap = new XrdAccCapability(path, xprivs);
                }
             lastcap->Add(currcap);
             lastcap = currcap;
            }

   // Check if all went well
   //
   if (NoGo) return -1;

   // Check if any capabilities were specified
   //
   if (!mycap.Next())
      {Eroute.Emsg("ConfigXeq", "no capabilities specified for", authid);
       return -1;
      }

   // Insert the capability into the appropriate table/list
   //
        if (domname)
           {if (!(ncp = new XrdAccCapName(authid, mycap.Next())))
               {Eroute.Emsg("ConfigXeq","unable to add id",authid); return -1;}
            if (tabs.E_List) tabs.E_List->Add(ncp);
               else tabs.D_List = ncp;
            tabs.E_List = ncp;
           }
   else if (anyuser) tabs.X_List = mycap.Next();
   else if (alluser) tabs.Z_List = mycap.Next();
   else    hp->Add(authid, mycap.Next());

   // All done
   //
   mycap.Add((XrdAccCapability *)0);
   return 1;
}
  
/******************************************************************************/
/*                          P r i v s C o n v e r t                           */
/******************************************************************************/
  
int XrdAccConfig::PrivsConvert(char *privs, XrdAccPrivCaps &ctab)
{
    int i = 0;
    XrdAccPrivs ptab[] = {XrdAccPriv_None, XrdAccPriv_None}; // Speed conversion here

    // Convert the privs
    //
    while(*privs)
         {switch((XrdAccPrivSpec)(*privs))
                {case    All_Priv:
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_All);
                            break;
                 case Delete_Priv:
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Delete);
                            break;
                 case Insert_Priv: 
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Insert);
                            break;
                 case   Lock_Priv: 
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Lock);
                            break;
                 case Lookup_Priv: 
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Lookup);
                            break;
                 case Rename_Priv: 
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Rename);
                            break;
                 case   Read_Priv: 
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Read);
                            break;
                 case  Write_Priv: 
                            ptab[i] = (XrdAccPrivs)(ptab[i]|XrdAccPriv_Write);
                            break;
                 case    Neg_Priv: if (i) return 0; i++;   break;
                 default:                 return 0;
                }
           privs++;
          }
     ctab.pprivs = ptab[0]; ctab.nprivs = ptab[1];
     return 1;
}
