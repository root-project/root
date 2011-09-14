// $Id$

const char *XrdSecpwdSrvAdminCVSID = "$Id$";
// ---------------------------------------------------------------------- //
//                                                                        //
//   Password file administration                                         //
//                                                                        //
//   Use this application to:                                             //
//                                                                        //
//      - Create / Modify a password file for servers under your          //
//        administration.                                                 //
//        Default location and name $(HOME)/.xrd/pwdadmin]                //
//                                                                        //
//          XrdSecpwdSrvAdmin [<file>]                                    //
//                                                                        //
//        NB: permissions must be such that the file is                   //
//            readable and writable by owner only, e.g. 0600              //
//                                                                        //
//                                                                        //
//      - Create / Modify a password file for servers enabled to verify   //
//        user passwords [default location and name $(HOME)/.xrd/pwduser] //
//                                                                        //
//          XrdSecpwdSrvAdmin -m user [<file>]                            //
//                                                                        //
//        NB: copy the file on the server machine if you are producing    //
//            it elsewhere; permissions must be such that the file is     //
//            writable by owner only, e.g. 0644                           //
//                                                                        //
//                                                                        //
//      - Create / Modify a autologin file                                //
//        [default location and name $(HOME)/.xrd/pwdnetrc]               //
//                                                                        //
//          XrdSecpwdSrvAdmin -m netrc [<file>]                           //
//                                                                        //
//        NB: permissions must be such that the file is                   //
//            readable and writable by owner only, e.g. 0600              //
//                                                                        //
//      - Create / Modify the file with server public cipher initiators   //
//        [default location and name $(HOME)/.xrd/pwdsrvpuk]              //
//                                                                        //
//          XrdSecpwdSrvAdmin -m srvpuk [<file>]                          //
//                                                                        //
//        NB: permissions must be such that the file is                   //
//            writable by owner only, e.g. 0644                           //
//                                                                        //
//                                                                        //
//  Author: G.Ganis, 2005                                                 //
// ---------------------------------------------------------------------- //
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <pwd.h>
#include <dirent.h>

#include <XrdOuc/XrdOucString.hh>

#include <XrdSut/XrdSutAux.hh>
#include <XrdSut/XrdSutPFEntry.hh>
#include <XrdSut/XrdSutPFile.hh>
#include <XrdSut/XrdSutRndm.hh>

#include <XrdCrypto/XrdCryptoCipher.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>

//
// enum
enum kModes {
   kM_undef = 0,
   kM_admin = 1,
   kM_user,
   kM_netrc,
   kM_srvpuk,
   kM_help
};
const char *gModesStr[] = {
   "kM_undef",
   "kM_admin",
   "kM_user",
   "kM_netrc",
   "kM_srvpuk",
   "kM_help"
};
enum kActions {
   kA_undef = 0,
   kA_add   = 1,
   kA_update,
   kA_read,
   kA_remove,
   kA_disable,
   kA_copy,
   kA_trim,
   kA_browse
};
const char *gActionsStr[] = {
   "kA_undef",
   "kA_add",
   "kA_update",
   "kA_read",
   "kA_remove",
   "kA_disable",
   "kA_copy",
   "kA_trim",
   "kA_browse"
};

//
// Globals 
int DebugON = 1;
XrdOucString DirRef   = "~/.xrd/";
XrdOucString AdminRef = "pwdadmin";
XrdOucString UserRef  = "pwduser";
XrdOucString NetRcRef = "pwdnetrc";
XrdOucString SrvPukRef= "pwdsrvpuk";
XrdOucString GenPwdRef= "/genpwd/";
XrdOucString GenPukRef= "/genpuk/";
XrdOucString IDTag    = "+++SrvID";
XrdOucString EmailTag = "+++SrvEmail";
XrdOucString HostTag  = "+++SrvHost";
XrdOucString PukTag   = "+++SrvPuk";
XrdOucString PwdFile  = "";
XrdOucString PukFile  = "/home/ganis/.xrd/genpuk/puk.07May2005-0849";
int          Mode     = kM_undef;
int          Action   = kA_undef;
int          NoBackup = 1;
XrdOucString NameTag  = "";
XrdOucString CopyTag  = "";
XrdOucString File     = "";
XrdOucString Path     = "";
XrdOucString Dir      = "";
XrdOucString SrvID    = "";
XrdOucString SrvName  = "";
XrdOucString Email    = "";
XrdOucString IterNum  = "";
bool         Backup   = 1;
bool         DontAsk  = 0;
bool         Force    = 0;
bool         Passwd   = 1;
bool         Change   = 1;
bool         Random   = 0;
bool         SavePw   = 1;
bool         SetID    = 0;
bool         SetEmail = 0;
bool         SetHost  = 0;
bool         Create   = 0;
bool         Confirm  = 1;
bool         Import   = 0;
bool         Hash     = 1;
bool         ChangePuk = 0;
bool         ChangePwd = 0;
bool         ExportPuk = 0;

#define NCRYPTMAX 10 // max number of crypto factories

XrdOucString DefCrypto = "ssl";
XrdOucString CryptList = "";
int          ncrypt    = 0; // number of available crypto factories
XrdOucString CryptMod[NCRYPTMAX] = {""}; // .. and their names
XrdCryptoCipher **RefCip = 0; // .. and their ciphers
XrdCryptoFactory  **CF = 0;
XrdCryptoKDFun_t KDFun = 0;
XrdCryptoKDFunLen_t KDFunLen = 0;

void Menu(int opt = 0);
int ParseArguments(int argc, char **argv);
void ParseCrypto();
bool CheckOption(XrdOucString opt, const char *ref, int &ival);
bool AddPassword(XrdSutPFEntry &ent, XrdOucString salt,
                 XrdOucString &ranpwd,
                 bool random, bool checkpw, bool &newpw);
bool AddPassword(XrdSutPFEntry &ent, bool &newpw, const char *pwd = 0);
void SavePasswd(XrdOucString tag, XrdOucString pwd, bool onetime);
bool ReadPasswd(XrdOucString &tag, XrdOucString &pwd, int &st);
bool ReadPuk(int &npuk, XrdOucString *tpuk, XrdOucString *puk);
int GeneratePuk();
bool SavePuk();
bool ReadPuk();
bool ExpPuk(const char *puk = 0, bool read = 1);
bool GetEntry(XrdSutPFile *ff, XrdOucString tag,
              XrdSutPFEntry &ent, bool &check);
bool AskConfirm(const char *msg1, bool defact, const char *msg2 = 0);
int LocateFactoryIndex(char *tag, int &id);

#define PRT(x) {cerr <<x <<endl;}
// Max number of attemps entreing a password
#define kMAXPWDATT 3

#define kMAXPUK 5
int nHostPuk;
XrdOucString TagHostPuk[kMAXPUK], HostPuk[kMAXPUK];

int main( int argc, char **argv )
{
   // Application for password file administration


   XrdSutPFEntry ent;
   XrdSutPFEntry *nent = 0;
   XrdOucString ans = "";
   XrdOucString email = "";
   XrdOucString uniqueid = "";
   XrdOucString tag = "";
   XrdOucString prompt = "Password: ";
   XrdOucString ranpwd = "";
   XrdOucString ImpPwd = "";
   XrdOucString salt = "";
   const char *pwdimp = 0;
   bool checkpwd = 0;
   bool newpw = 1;
   bool check = 0;
   int nr = 0, nm = 0;
   int i = 0;
   int entst = 0;

   // Parse arguments
   if (ParseArguments(argc,argv)) {
      exit(0);
   }
   ParseCrypto();

   // Set trace options
   XrdSutSetTrace(sutTRACE_Debug);

   // Attach to file
   kXR_int32 openmode = (Create) ? kPFEcreate : 0;
   XrdSutPFile ff(File.c_str(), openmode);
   if (!ff.IsValid() && ff.LastError() == kPFErrNoFile) {
      prompt = "Create file ";
      prompt += File;
      if (DontAsk || AskConfirm(prompt.c_str(),0)) {
         if (Mode == kM_user || Mode == kM_srvpuk)
            ff.Init(File.c_str(), kPFEcreate, 0644);
         else
            ff.Init(File.c_str(), kPFEcreate);
      }
      if (!ff.IsValid())
         exit(1);
      if (Mode == kM_admin || Mode == kM_user) {
         if (SrvID.length() <= 0) {
            if (!DontAsk && AskConfirm("Would you like to enter a server ID? ",1)) {
               XrdSutGetLine(SrvID,"Enter ID (max 32 chars): ");
               if (SrvID.length() > 32)
                  SrvID.erase(32);
            } else {
               PRT("Server ID will be generated randomly. It can be changed");
               PRT("at any time with 'add -srvID <ID>'.");
               //
               // Set random ID
               XrdSutRndm::Init();
               XrdSutRndm::GetString(1,8,SrvID);
               //
               // Add local user name
               struct passwd *pw = getpwuid(getuid());
               if (pw) {
                  SrvID.insert(':',0);
                  SrvID.insert(pw->pw_name,0);
               }
            }
         } else if (DontAsk) {
            // This is a force creation where no prompt request can be answered
            SetID = 0;
         }
         PRT("Server ID: " << SrvID);
         if (SrvID.length() > 0) {
            //
            // Fill entry
            ent.SetName(IDTag.c_str());
            ent.status = kPFE_special;
            ent.cnt    = 1;
            ent.buf1.SetBuf(SrvID.c_str(),SrvID.length()+1);
            //
            // Write entry
            ent.mtime = time(0);
            ff.WriteEntry(ent);
            PRT(" File successfully created with server ID set to: "
                  <<SrvID.c_str());
         }
         // Generate srvpuk for admin
         if (Mode == kM_admin) {

            int ncf = GeneratePuk();
            if (ncf != ncrypt)
               PRT("// Could generate ref ciphers for all the factories");

            // Update file
            for ( i = 0; i < ncrypt; i++ ) {
               if (RefCip[i]) {
                  //
                  // Build tag
                  tag = PukTag + '_';
                  tag += CF[i]->ID();
                  //
                  // Serialize in a buffer
                  XrdSutBucket *bck = RefCip[i]->AsBucket();
                  if (bck) {
                     //
                     // Prepare Entry
                     ent.SetName(tag.c_str());
                     ent.status = kPFE_special;
                     ent.cnt    = 2;  // protected
                     ent.buf1.SetBuf(bck->buffer,bck->size);
                     //
                     // Write entry
                     ent.mtime = time(0);
                     ff.WriteEntry(ent);
                     PRT(" Server Puk saved for crypto: "<<CF[i]->Name());
                     delete bck;
                     bck = 0;
                  }
               }
            }
            //
            // Backup also on separate file
            if (!SavePuk()) {
               PRT("// Problems with puk backup ");
            }
         }
      } else {
         PRT(" File successfully created ");
      }
   }

   // If admin, check for special entries
   // (Server Unique ID, Email, Host name)
   if (Mode == kM_admin) {
      //
      // Ref ciphers
      ent.Reset();
      nm = ff.SearchEntries(PukTag.c_str(),0);
      if (nm) {
         int *ofs = new int[nm];
         ff.SearchEntries(PukTag.c_str(),0,ofs,nm);
         for ( i = 0; i < nm ; i++) {
            nr = ff.ReadEntry(ofs[i],ent);
            if (nr > 0) {
               XrdSutBucket bck;
               bck.SetBuf(ent.buf1.buf,ent.buf1.len);
               // Locate factory ID
               int id = 0;
               int ii = LocateFactoryIndex(ent.name, id);
               if (ii < 0) {
                  PRT("// Factory ID not found: corruption ?");
                  exit(1);
               }
               if (!(RefCip[i] = CF[ii]->Cipher(&bck))) {
                  PRT("// Could not instantiate cipher for factory "<<CF[ii]->Name());
                  exit(1);
               }
            }
         }
      } else {
         PRT("// Ref puk ciphers not found: corruption ?");
         exit(1);
      }


      if (ff.ReadEntry(IDTag.c_str(),ent) <= 0 && !SetID) {
         PRT(" Unique ID missing: 'add -srvID' to set it");
      } else if (!SetID) {
         SrvID.insert(ent.buf1.buf,0,ent.buf1.len);
      }
      //
      // Unique ID
      ent.Reset();
      if (ff.ReadEntry(IDTag.c_str(),ent) <= 0 && !SetID) {
         PRT(" Unique ID missing: 'add -srvID' to set it");
      } else if (!SetID) {
         SrvID.insert(ent.buf1.buf,0,ent.buf1.len);
      }
      //
      // Email
      ent.Reset();
      if (ff.ReadEntry(EmailTag.c_str(),ent) <= 0 && !SetEmail) {
         PRT(" Contact E-mail not set: 'add -email <email>' to set it");
      } else if (!SetEmail) {
         Email.insert(ent.buf1.buf,0,ent.buf1.len);
      }
      //
      // Server Host name 
      ent.Reset();
      if (ff.ReadEntry(HostTag.c_str(),ent) <= 0 && !SetHost) {
         PRT(" Local host name not set: 'add -host <host>' to set it");
      } else if (!SetHost) {
         SrvName.insert(ent.buf1.buf,0,ent.buf1.len);
      }
   }

   switch (Action) {
   case kA_update:
      //
      // Like 'add', forcing write
      Force = 1;
   case kA_add:
      //
      // Add / Update entry
      //
      // If admin, check first if we are required to update/create
      // some special entry (Server Unique ID, Email, Host Name)
      if (Mode == kM_admin) {
         //
         // Export current Server PUK
         if (ExportPuk) {
            if (!ExpPuk()) {
               PRT("// Could not export public keys");
            }
            //
            // We are done
            break;
         }
         //
         // Server PUK
         ent.Reset();
         if (ChangePuk) {
            if (!DontAsk && !AskConfirm("Override server PUK?",0,0))
               break;
            //
            // If we are given a file name, try import from the file
            if (Import && PukFile.length() > 0) {
               if (!ReadPuk()) {
                  PRT("// Problem importing puks from "<<PukFile<<
                      " - exit ");
                  break;
               }
            } else {
               // Generate new puks
               if (GeneratePuk() != ncrypt) {
                  PRT("// Could not generate ref ciphers for all the factories");
                  break;
               }
            }
            //
            // Backup also on separate file
            if (!SavePuk()) {
               PRT("// Problems with puk backup ");
            }
            //
            // Now shift up the old one(s)
            nm = ff.SearchEntries(PukTag.c_str(),0);
            if (nm) {
               PRT("// Found "<<nm<<" entries for tag '"<<PukTag.c_str()<<
                           "' in file: "<<ff.Name());
               //
               // Book vector for offsets
               int *ofs = new int[nm];
               //
               // Get number of entries related
               ff.SearchEntries(PukTag.c_str(),0,ofs,nm);
               //
               // Read entries now
               for ( i = 0; i < nm ; i++) {
                  nr = ff.ReadEntry(ofs[i],ent);
                  if (nr > 0) {
                     //
                     // Locate factory ID
                     int id;
                     int j = LocateFactoryIndex(ent.name,id);
                     if (j < 0) break;
                     // Serialize in a buffer
                     XrdSutBucket *bck = RefCip[j]->AsBucket();
                     if (bck) {
                        // Shift up buffer content (buf 4 is removed)
                        if (ent.buf4.buf)
                           delete[] ent.buf4.buf;
                        ent.buf4.buf = ent.buf3.buf;
                        ent.buf4.len = ent.buf3.len;
                        ent.buf3.buf = ent.buf2.buf;
                        ent.buf3.len = ent.buf2.len;
                        ent.buf2.buf = ent.buf1.buf;
                        ent.buf2.len = ent.buf1.len;
                        // fill buf 1 with new puk
                        ent.buf1.SetBuf(bck->buffer,bck->size);
                        //
                        // Write entry
                        ent.mtime = time(0);
                        ff.WriteEntry(ent);
                        PRT(" Server Puk updated for crypto: "<<CF[i]->Name());
                        delete bck;
                        bck = 0;
                     }
                     //
                     // Flag user entries
                     char stag[4];
                     sprintf(stag,"*_%d",id);
                     int nofs = ff.SearchEntries(stag,2);
                     if (nofs > 0) {
                        int *uofs = new int[nofs];
                        ff.SearchEntries(stag,2,uofs,nofs);
                        XrdSutPFEntry uent;
                        int k = 0, nnr = 0;
                        for (; k < nofs; k++) {
                           uent.Reset();
                           nnr = ff.ReadEntry(uofs[k],uent);
                           if (nnr > 0 && !strstr(uent.name,PukTag.c_str())) {
                              char c = 0;
                              if (uent.buf4.buf) {
                                 c = *(uent.buf4.buf);
                                 c++;
                                 if (c > 4)
				   c = 1;
                                 *(uent.buf4.buf) = c;
                              } else {
                                 uent.buf4.buf = new char[1];
                                 uent.buf4.len = 1;
                                 *(uent.buf4.buf) = 2;
                              }
                              // Write entry
                              uent.mtime = time(0);
                              ff.WriteEntry(uent);
                           }
                        }
                     }
                  } else {
                     PRT("// warning: problems reading entry: corruption?");
                     break;
                  }
               }
            } else {
               PRT("// WARNING: No entry for tag '"<<PukTag.c_str()<<
                   "' found in file: "<<ff.Name()<<" : corruption? ");
               break;
            }
         }
         //
         // Server Unique ID
         ent.Reset();
         if (SetID) {
            if (!GetEntry(&ff,IDTag,ent,check)) {
               if (!check || AskConfirm("Override server ID?",0,
                                        "This may cause inconveniences"
                                        " to clients")) {
                  //
                  // Prepare Entry
                  ent.SetName(IDTag.c_str());
                  ent.status = kPFE_special;
                  ent.cnt    = 1;
                  ent.buf1.SetBuf(SrvID.c_str(),SrvID.length()+1);
                  //
                  // Write entry
                  ent.mtime = time(0);
                  ff.WriteEntry(ent);
                  PRT(" Server ID set to: "<<SrvID.c_str());
               }
            }
         }
         //
         // Email
         ent.Reset();
         if (SetEmail) {
            if (!GetEntry(&ff,EmailTag,ent,check)) {
               if (!check || AskConfirm("Override contact e-mail"
                                        " address?",0)) {
                  //
                  // Prepare Entry
                  ent.SetName(EmailTag.c_str());
                  ent.status = kPFE_special;
                  ent.cnt    = 1;
                  ent.buf1.SetBuf(Email.c_str(),Email.length()+1);
                  //
                  // Write entry
                  ent.mtime = time(0);
                  ff.WriteEntry(ent);
                  PRT(" Contact e-mail set to: "<<Email.c_str());
               }
            }
         }
         //
         // Server host name
         ent.Reset();
         if (SetHost) {
            if (!GetEntry(&ff,HostTag,ent,check)) {
               if (!check || AskConfirm("Override server host name?",0)) {
                  //
                  // Prepare Entry
                  ent.SetName(HostTag.c_str());
                  ent.status = kPFE_special;
                  ent.cnt    = 1;
                  ent.buf1.SetBuf(SrvName.c_str(),SrvName.length()+1);
                  //
                  // Write entry
                  ent.mtime = time(0);
                  ff.WriteEntry(ent);
                  PRT(" Server host name set to: "<<SrvName.c_str());
               }
            }
         }
 
      }
      //
      // If import mode for read info from file
      if (Mode == kM_srvpuk) {
         if (!Import) {
            PRT("// Updating the server puk file requires a file with "<<
                "the keys received by the server administrator:");
            PRT("// rerun with option '-import <file_with_keys>' ");
            break;
         }
         if (!ReadPuk(nHostPuk,TagHostPuk,HostPuk))
            break;
         //
         // Now we loop over tags
         for (i = 0; i < nHostPuk; i++) {
            // Check if not already existing
            ent.Reset();
            if (GetEntry(&ff,TagHostPuk[i],ent,check)) {
               break;
            }
            // Fill in new puk
            ent.buf1.SetBuf(HostPuk[i].c_str(),HostPuk[i].length()+1);
            // Write entry
            ent.mtime = time(0);
            ff.WriteEntry(ent);
            if (check) {
               PRT("// Server puk "<<TagHostPuk[i]<<" updated");
            } else {
               PRT("// Server puk "<<TagHostPuk[i]<<" added");
            }
         }
         //
         // Browse new content
         ff.Browse();
         //
         // We are done
         break;
      }
      //
      // If import mode for read info from file
      if (Mode == kM_netrc) {
         if (Import) {
            if (!ReadPasswd(NameTag,ImpPwd,entst))
               break;
            pwdimp = ImpPwd.c_str();;
         }
         // Special treatment for non-hashed passwords (provided
         // to allow store info for crypt-like credentials)
         if (!Hash) {
            // Check if not already existing
            ent.Reset();
            if (GetEntry(&ff,NameTag,ent,checkpwd)) {
               break;
            }
            // Reset status and cnt
            ent.status = entst;
            ent.cnt    = 0;
            //
            // Fill with password
            if (!AddPassword(ent, newpw, pwdimp)) {
               PRT("Error creating new password: "<<gModesStr[Mode]);
               break;
            }
            //
            // Save (or update) entry
            ent.mtime = time(0);
            ff.WriteEntry(ent);
            PRT(" Entry for tag '"<<NameTag<<
                "' created / updated");
            // We are done
            break;
         }
      }
      //
      // Now we need a name tag
      if (!NameTag.length()) break;
      //
      // Ask confirmation, if required
      prompt = "Adding/Updating entry for tag: ";
      prompt += NameTag;
      if (!DontAsk && !AskConfirm("Do you want to continue?",0,prompt.c_str()))
         break;
      //
      // Normal operations
      KDFun = 0;
      KDFunLen = 0;
      newpw = 1;
      //
      // New salt (random machinery init only once)
      if (Mode != kM_netrc) {
         XrdSutRndm::Init();
         XrdSutRndm::GetString(3,8,salt);
         if (IterNum.length() > 0) {
            // Insert non default iteration number in salt
            salt.insert(IterNum,0);         
         }
      }
      //
      for ( i = 0; i < ncrypt; i++ ) {
         // Get hook to crypto factory
         CF[i] = XrdCryptoFactory::GetCryptoFactory(CryptMod[i].c_str());
         if (!CF[i]) {
            PRT("Hook for crypto factory undefined: "<<CryptMod[i].c_str());
            break;
         }
         //
         // Get one-way hash function
         KDFun = CF[i]->KDFun();
         KDFunLen = CF[i]->KDFunLen();
         if (!KDFun || !KDFunLen) {
            PRT("Error resolving one-way hash functions ");
            break;
         }
         //
         // Build tag
         tag = NameTag + '_';
         tag += CF[i]->ID();
         // Check if not already existing
         ent.Reset();
         if (GetEntry(&ff,tag,ent,checkpwd)) {
            break;
         }
         if (Mode == kM_netrc) {
            // If just a request for password change not much to do
            if (ChangePwd) {
               if (!checkpwd)
                  break;
               else
                  // Update the status
                  ent.status = kPFE_onetime;
            } else {
               // Reset status and cnt
               if (pwdimp)
                  ent.status = entst;
               else
                  ent.status = kPFE_ok;
               ent.cnt    = 0;
               //
               // Fill with password
               if (!AddPassword(ent, newpw, pwdimp)) {
                  PRT("Error creating new password: "<<gModesStr[Mode]);
                  break;
               }
            }
         } else {
            // Reset cnt
            ent.cnt    = 0;
            if (Passwd) {
               // Set status
               ent.status = Change ? kPFE_onetime : kPFE_ok;
               //
               // Fill with password
               if (!AddPassword(ent, salt, ranpwd, Random, checkpwd, newpw)) {
                  PRT("Error creating new password: "<<gModesStr[Mode]);
                  break;
               }
            } else {
               ent.buf1.SetBuf();
               ent.buf2.SetBuf();
               ent.buf3.SetBuf();
               ent.buf4.SetBuf();
               // Just enable entry
               ent.status = kPFE_allowed;
            }
         }
         //
         // Save (or update) entry
         ent.mtime = time(0);
         ff.WriteEntry(ent);
         PRT(" Entry for tag '"<<tag.c_str()<<
             "' created / updated");
      }
      //
      // Save password, if requested
      if (SavePw)
         SavePasswd(NameTag, ranpwd, Change);

      // Browse the new status
      ff.Browse();
      break;

   case kA_read:
      //
      // Get number of entries related
      nm = ff.SearchEntries(NameTag.c_str(),0);
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      PRT("//");
      if (nm) {
         PRT("// Found "<<nm<<" entries for tag '"<<NameTag.c_str()<<
                     "' in file: "<<ff.Name());
         //
         // Book vector for offsets
         int *ofs = new int[nm];
         //
         // Get number of entries related
         ff.SearchEntries(NameTag.c_str(),0,ofs,nm);
         //
         // Read entries now
         for ( i = 0; i < nm ; i++) {
            nr = ff.ReadEntry(ofs[i],ent);
            if (nr > 0) {
               PRT("// #:"<<i+1<<"  "<<ent.AsString());
            } else {
               PRT("// Entry for ofs "<<ofs[i]<<
                     " not found in file: "<<ff.Name());
            }
         }
      } else {
         PRT("// No entry for tag '"<<NameTag.c_str()<<
               "' found in file: "<<ff.Name());
      }
      PRT("//");
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      break;

   case kA_remove:
      //
      // Ask confirmation, if required
      prompt = "Removing entry for tag: ";
      prompt += NameTag;
      if (!DontAsk && !AskConfirm("Do you want to continue?",0,prompt.c_str()))
         break;
      //
      // Get number of entries related
      nm = ff.SearchEntries(NameTag.c_str(),0);
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      PRT("//");
      if (nm) {
         PRT("// Found "<<nm<<" entries for tag '"<<NameTag.c_str()<<
                     "' in file: "<<ff.Name());
         //
         // Book vector for offsets
         int *ofs = new int[nm];
         //
         // Get number of entries related
         ff.SearchEntries(NameTag.c_str(),0,ofs,nm);
         //
         // Read entries now
         for ( i = 0; i < nm ; i++) {
            if (ff.RemoveEntry(ofs[i]) == 0) {
               PRT("// Entry for tag '"<<NameTag.c_str()<<
                     "' removed from file: "<<ff.Name());
            } else {
               PRT("// Entry for tag '"<<NameTag.c_str()<<
                     "' not found in file: "<<ff.Name());
            }
         }
      } else {
         PRT("// No entry for tag '"<<NameTag.c_str()<<
               "' found in file: "<<ff.Name());
      }
      PRT("//");
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      break;

   case kA_disable:
      //
      // Ask confirmation, if required
      prompt = "Disabling entry for tag: ";
      prompt += NameTag;
      if (!DontAsk && !AskConfirm("Do you want to continue?",0,prompt.c_str()))
         break;
      //
      // Get number of entries related
      nm = ff.SearchEntries(NameTag.c_str(),0);
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      PRT("//");
      if (nm) {
         PRT("// Found "<<nm<<" entries for tag '"<<NameTag.c_str()<<
                     "' in file: "<<ff.Name());
         //
         // Book vector for offsets
         int *ofs = new int[nm];
         //
         // Get number of entries related
         ff.SearchEntries(NameTag.c_str(),0,ofs,nm);
         //
         // Read entries now
         for ( i = 0; i < nm ; i++) {
            nr = ff.ReadEntry(ofs[i],ent);
            if (nr > 0) {
               // Disable entry
               ent.status = kPFE_disabled;
               ent.cnt = 0;
               ent.buf1.SetBuf();
               ent.buf2.SetBuf();
               ent.buf3.SetBuf();
               ent.buf4.SetBuf();
               // Save (or update) entry
               ent.mtime = time(0);
               ff.WriteEntry(ent);
               PRT("// Entry for tag '"<<ent.name<<
                   "' disabled");
            } else {
               PRT("// Entry for ofs "<<ofs[i]<<
                     " not found in file: "<<ff.Name());
            }
         }
      } else {
         PRT("// No entry for tag '"<<NameTag.c_str()<<
               "' found in file: "<<ff.Name());
      }
      PRT("//");
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      break;

   case kA_copy:
      //
      // Ask confirmation, if required
      prompt = "Copying entry for tag: ";
      prompt += NameTag;
      prompt += " into tag: ";
      prompt += CopyTag;
      if (!DontAsk && !AskConfirm("Do you want to continue?",0,prompt.c_str()))
         break;
      //
      // Ready entry
      if (ff.ReadEntry(NameTag.c_str(),ent) <= 0) {
         PRT("Entry to copy not found missing");
         break;
      }
      //
      // Prepare New Entry
      nent = new XrdSutPFEntry(ent);
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      PRT("//");
      if (nent) {
         nent->SetName(CopyTag.c_str());
         //
         // Write entry
         nent->mtime = time(0);
         ff.WriteEntry(*nent);
         PRT("// Entry for tag '"<<nent->name<<
             "' created");
         delete nent;
      } else {
         PRT("// Cannot create new entry: out of memory");
         break;
      }
      PRT("//");
      PRT("//-----------------------------------------------------"
                                             "--------------------//");
      break;

   case kA_trim:
      //
      // Trim the file
      ff.Trim();

   case kA_browse:
   default:
      //
      // Browse
      ff.Browse();
   }

   exit(0);
}


void Menu(int opt)
{
   // Print the menu
   // Options:          0        intro w/  head/tail
   //                   1        intro w/o head/tail
   //                   2        keywords

   // Head
   if (opt == 0) {
     PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
     PRT("+                                                          +");
     PRT("+                  x r d p w d a d m i n                   +");
     PRT("+                                                          +");
     PRT("+                Administration of pwd files               +");
   }

   // Intro
   if (opt <= 1) {
     PRT("+                                                          +");
     PRT("+  Syntax:                                                 +");
     PRT("+                                                          +");
     PRT("+  xrdpwdadmin [-h] [-m <mode>] [options]                  +");
     PRT("+                                                          +");
     PRT("+   -h   display this menu                                 +");
     PRT("+                                                          +");
     PRT("+   -m   choose mode (admin, user, netrc, srvpuk) [admin]  +");
     PRT("+                                                          +");
     PRT("+        admin:                                            +");
     PRT("+        create / modify the main file used by servers     +");
     PRT("+        started from this account to validate clients     +");
     PRT("+        credentials. Default location and name:           +");
     PRT("+                 $(HOME)/.xrd/pwdadmin                    +");
     PRT("+                                                          +");
     PRT("+        NB: file must readable and writable by owner      +");
     PRT("+            only e.g. 0600                                +");
     PRT("+                                                          +");
     PRT("+        user:                                             +");
     PRT("+        create / modify local file used by servers        +");
     PRT("+        to validate this user credentials.                +");
     PRT("+        Default location and name:                        +");
     PRT("+                 $(HOME)/.xrd/pwduser                     +");
     PRT("+                                                          +");
     PRT("+        NB: the file must be copied on the server machine +");
     PRT("+            if produced elsewhere; file must be writable  +");
     PRT("+            by the owner only, e.g. 0644                  +");
     PRT("+                                                          +");
     PRT("+        netrc:                                            +");
     PRT("+        create / modify local autologin file              +");
     PRT("+        Default location and name:                        +");
     PRT("+                 $(HOME)/.xrd/pwdnetrc                    +");
     PRT("+                                                          +");
     PRT("+        NB: file must readable and writable by owner      +");
     PRT("+            only e.g. 0600                                +");
     PRT("+                                                          +");
     PRT("+        srvpuk:                                           +");
     PRT("+        create / modify local file with known server      +");
     PRT("+        public cipher initializers.                       +");
     PRT("+        Default location and name:                        +");
     PRT("+                 $(HOME)/.xrd/pwdsrvpuk                   +");
     PRT("+                                                          +");
     PRT("+        NB: file must be writable by the owner only       +");
     PRT("+            e.g. 0644                                     +");
   }

   // Intro
   if (opt <= 2) {
     PRT("+                                                          +");
     PRT("+  Options:                                                +");
     PRT("+                                                          +");
     PRT("+   add <name> [-[no]force] [-[no]random] [-[no]savepw]    +");
     PRT("+      add entry with tag <name>; the application prompts  +");
     PRT("+      for the password                                    +");
     PRT("+                                                          +");
     PRT("+   add <name> -import <pwd_file>                          +");
     PRT("+      add entry with tag <name> importing the pwd from    +");
     PRT("+      the file send by the server administrator           +");
     PRT("+      [netrc only]                                        +");
     PRT("+                                                          +");
     PRT("+   add -import <srvkey_file>                              +");
     PRT("+      add new server key importing the key from           +");
     PRT("+      the file send by the server administrator           +");
     PRT("+      [srvpuk only]                                       +");
     PRT("+                                                          +");
     PRT("+   update <name> [options]                                +");
     PRT("+      equivalent to 'add -force'                          +");
     PRT("+                                                          +");
     PRT("+   read <name>                                            +");
     PRT("+      list some information of entry associated with tag  +");
     PRT("+      <name> (status, count, date of last change, buffer  +");
     PRT("+      lengths); buffer contents not listed                +");
     PRT("+                                                          +");
     PRT("+   remove <name>                                          +");
     PRT("+      Make entry associated with tag <name> inactive      +");
     PRT("+      (Spce is recovered during next trim operation)      +");
     PRT("+                                                          +");
     PRT("+   copy <name> <newname>                                  +");
     PRT("+      Create new entry with tag <newname> and content of  +");
     PRT("+      existing entry with tag <name>                      +");
     PRT("+                                                          +");
     PRT("+   trim [-nobackup]                                       +");
     PRT("+      Trim the file content eliminating all the inactive  +");
     PRT("+      entries; a backup is created in <file>.bak unless   +");
     PRT("+      the option '-nobackup' is specified                 +");
     PRT("+                                                          +");
     PRT("+   browse                                                 +");
     PRT("+      list a table about the file content                 +");
   }

   // Intro
   if (opt <= 3) {
     PRT("+                                                          +");
     PRT("+   -dontask                                               +");
     PRT("+      do not prompt for questions: when in doubt use      +");
     PRT("+      defaults or fail                                    +");
     PRT("+      [default: ask]                                      +");
     PRT("+   -force                                                 +");
     PRT("+      overwrite entry if it exists already                +");
     PRT("+      [default: do not overwrite]                         +");
     PRT("+   -[no]change                                            +");
     PRT("+      do [not] require user to change info on first use   +");
     PRT("+      [default: admin: change / user: no change           +");
     PRT("+   -crypto [-]<crypt1>|[-]<crypt2>|...                    +");
     PRT("+      create information for the given crypto modules     +");
     PRT("+      ('|' separated list) in addition to default ones    +");
     PRT("+      (normally ssl and local); use '-' in front to avoid +");
     PRT("+      avoid creating a entry for a module; one entry is   +");
     PRT("+      for each module with effective tag of the form      +");
     PRT("+      name_<cryptoID> [default list: ssl]                 +");
     PRT("+                 [default: create backup]                 +");
   }

   // Tail
   PRT("+                                                          +");
   PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
}

int ParseArguments(int argc, char **argv)
{
   // Parse application arguments filling relevant global variables
   bool changeset = 0;
   bool randomset = 0;
   bool savepwset = 0;
   bool randomid  = 0;

   // Number of arguments
   if (argc < 0 || !argv[0]) {
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      PRT("+ Insufficient number or arguments!                        +");
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      // Print main menu
      Menu(0);
      return 1;
   }
   --argc;
   ++argv;

   //
   // Loop over arguments
   while ((argc >= 0) && (*argv)) {

      XrdOucString opt = "";
      int ival = -1;
      if(*(argv)[0] == '-') {

         opt = *argv;
         opt.erase("-"); 
         if (CheckOption(opt,"m",ival)) {
            if (Mode != kM_undef) {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Only one valid '-m' option allowed: ignoring             +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               --argc;
               ++argv;
               if (argc >= 0 && (*argv && *(argv)[0] == '-')) {
                  argc++;
                  argv--;
               }
            }
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               XrdOucString mode = *argv;
               if (CheckOption(mode,"admin",ival)) {
                  Mode = kM_admin;
               } else if (CheckOption(mode,"user",ival)) {
                  Mode = kM_user;
               } else if (CheckOption(mode,"netrc",ival)) {
                  Mode = kM_netrc;
               } else if (CheckOption(mode,"srvpuk",ival)) {
                  Mode = kM_srvpuk;
               } else if (CheckOption(mode,"help",ival)) {
                  Mode = kM_help;
               } else {
                  PRT("++++++++++++++++++++++++++++++++++++++"
                                                 "++++++++++++++++++++++");
                  PRT("+ Ignoring unrecognized more: "<<mode.c_str());
                  PRT("++++++++++++++++++++++++++++++++++++++"
                                                 "++++++++++++++++++++++");
               }
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-m' requires {admin,user,netrc,srvpuk}: ignoring +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"h",ival) ||
                    CheckOption(opt,"help",ival) ||
                    CheckOption(opt,"menu",ival)) {
            Mode = kM_help;
         } else if (CheckOption(opt,"f",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               Path = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-f' requires a file or directory name: ignoring  +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"dontask",ival)) {
            DontAsk = ival;
         } else if (CheckOption(opt,"force",ival)) {
            Force = ival;
         } else if (CheckOption(opt,"change",ival)) {
            Change = ival;
            changeset = 1;
         } else if (CheckOption(opt,"passwd",ival)) {
            Passwd = ival;
         } else if (CheckOption(opt,"backup",ival)) {
            Backup = ival;
         } else if (CheckOption(opt,"random",ival)) {
            Random = ival;
            randomset = 1;
         } else if (CheckOption(opt,"savepw",ival)) {
            SavePw = ival;
            savepwset = 1;
         } else if (CheckOption(opt,"confirm",ival)) {
            Confirm = ival;
         } else if (CheckOption(opt,"create",ival)) {
            Create = ival;
         } else if (CheckOption(opt,"hash",ival)) {
            Hash = ival;
         } else if (CheckOption(opt,"changepuk",ival)) {
            ChangePuk = ival;
         } else if (CheckOption(opt,"changepwd",ival)) {
            ChangePwd = ival;
         } else if (CheckOption(opt,"exportpuk",ival)) {
            ExportPuk = ival;
         } else if (CheckOption(opt,"iternum",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               int iter = strtol(*argv,0,10);
               if (iter > 0 && errno != ERANGE) {
                  IterNum = "$$";
                  IterNum += *argv;
                  IterNum += "$";
               } else {
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  PRT("+ Option '-iternum' requires a positive number: ignoring   +");
                  PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                  argc++;
                  argv--;
               }
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-iternum' requires a positive number: ignoring   +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"crypto",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               CryptList = *argv;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-crypto' requires a list of modules: ignoring    +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"import",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               if (Mode == kM_netrc) {
                  PwdFile = *argv;
               } else {
                  PukFile = *argv;
               }
               Import = 1;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-import' requires a file name: ignoring          +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"srvID",ival)) {
            --argc;
            ++argv;
            SetID = 1;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               SrvID = *argv;
            } else {
               SrvID = "";
               randomid = 1;
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"email",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               Email = *argv;
               SetEmail = 1;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-email' requires an email string: ignoring       +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"host",ival)) {
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               SrvName = *argv;
               SetHost = 1;
            } else {
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               PRT("+ Option '-host' requires the local host name: ignoring    +");
               PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
               argc++;
               argv--;
            }
         } else {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Ignoring unrecognized option: "<<*argv);
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         }

      } else {
         //
         // Action keyword
         opt = *argv;
         int iad = -1, iup = -1, ird = -1, irm = -1, idi = -1, icp = -1; 
         if (CheckOption(opt,"add",iad) || CheckOption(opt,"update",iup) ||
             CheckOption(opt,"read",ird) || CheckOption(opt,"remove",irm) ||
             CheckOption(opt,"disable",idi) || CheckOption(opt,"copy",icp)) {
            Action = (Action == kA_undef && iad == 1) ? kA_add : Action;
            Action = (Action == kA_undef && iup == 1) ? kA_update : Action;
            Action = (Action == kA_undef && ird == 1) ? kA_read : Action;
            Action = (Action == kA_undef && irm == 1) ? kA_remove : Action;
            Action = (Action == kA_undef && idi == 1) ? kA_disable : Action;
            Action = (Action == kA_undef && icp == 1) ? kA_copy : Action;
            --argc;
            ++argv;
            if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
               NameTag = *argv;
               if (icp == 1) {
                  --argc;
                  ++argv;
                  if (argc >= 0 && (*argv && *(argv)[0] != '-')) {
                     CopyTag = *argv;
                  } else {
                     PRT("+++++++++++++++++++++++++++++++++++++++++"
                                               "+++++++++++++++++++");
                     PRT("+ 'copy': missing destination tag: ignoring"
                                                 "                +"); 
                     PRT("+++++++++++++++++++++++++++++++++++++++++"
                                               "+++++++++++++++++++");
                     CopyTag = "";
                     argc++;
                     argv--;
                  }
               }
            } else {
               NameTag = "";
               argc++;
               argv--;
            }
         } else if (CheckOption(opt,"trim",ival)) {
            Action = kA_trim;
         } else if (CheckOption(opt,"browse",ival)) {
            Action = kA_browse;
         } else {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Ignoring unrecognized keyword action: "<<opt.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         }
      }
      --argc;
      ++argv;
   }

   //
   // Default mode 'admin'
   Mode = (Mode == 0) ? kM_admin : Mode;

   //
   // If help mode, print menu and exit
   if (Mode == kM_help) {
      // Print main menu
      Menu(0);
      return 1;
   }

   //
   // Some action need a tag name
   bool special = SetID || SetEmail || SetHost || ChangePuk || ExportPuk;
   if (Action == kA_add || Action == kA_update ||
       Action == kA_read || Action == kA_remove) {
      if (!special && !NameTag.length() &&!Import) {
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         PRT("+ Specified action requires a tag: "<<
                gActionsStr[Action]);
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         Menu(2);
         return 1;
      }
   }

   //
   // If user mode, check if NameTag contains the local user
   // name: if not, warn the user about possible problems with
   // servers ignoring this kind of entries for users files
   if (Mode == kM_admin && SetID) {
      if (randomid) {
         // Set random ID
         XrdSutRndm::Init();
         XrdSutRndm::GetString(1,8,SrvID);
         // Add local user name
         struct passwd *pw = getpwuid(getuid());
         if (pw) {
            SrvID.insert(':',0);
            SrvID.insert(pw->pw_name,0);
         } else { 
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ WARNING: could not get local user info for srv ID        +");
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         }
      } else {
         if (SrvID.length() > 32) {
            SrvID.erase(32);
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ WARNING: srv ID too long: truncating to 32 chars: "
                   <<SrvID.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         }
      }
   }

   //
   // Setting a non default iteration number is only allowed
   // in admin or user mode, to avoid potential inconsistencies
   if (IterNum.length() > 0 && (Mode != kM_admin && Mode != kM_user)) {
      IterNum = "";
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      PRT("+ WARNING: ignore iter num change request (not admin/user) +");
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
   }

   //
   // Requesting a password change only makes sense in netrc mode
   if (ChangePwd && Mode != kM_netrc) {
      ChangePwd = 0;
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
      PRT("+ WARNING: ignore password change request (not netrc)      +");
      PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
   }

   //
   // If user mode, check if NameTag contains the local user
   // name: if not, warn the user about possible problems with
   // servers ignoring this kind of entries for users files
   if (Mode == kM_user && NameTag.length()) {
      struct passwd *pw = getpwuid(getuid());
      if (pw) {
         XrdOucString locusr = pw->pw_name;
         if (NameTag != locusr) {
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ WARNING: name tag does not match local user name: ");
            PRT("+          "<<NameTag.c_str()<<"  "<<locusr.c_str());
            PRT("+ Some servers may ignore this entry ");
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            if (Action == kA_add)
               Confirm = 1; 
         }
      }
   }

   //
   // Default action 'browse', except for specials
   Action = (Action == kA_undef && special) ? kA_add : Action;
   Action = (Action == kA_undef) ? kA_browse : Action;

   //
   // Set defaults according to mode, if required
   if (Mode == kM_admin) {
      Change = (changeset) ? Change : 1;
      Random = (randomset) ? Random : 1;
      SavePw = (savepwset) ? SavePw : 1;
   } else {
      Change = (changeset) ? Change : 0;
      Random = (randomset) ? Random : 0;
      SavePw = (savepwset) ? SavePw : 0;
   }

   //
   // 'Create' can be active only for 'add' or 'update'
   Create = (Action == kA_add || Action == kA_update) ? Create : 0;

   //
   // If defined, check nature of Path (if it exists)
   if (Path.length()) {
      //
      // Expand Path
      XrdSutExpand(Path);
      // Get info
      struct stat st;
      if (stat(Path.c_str(),&st) == 0) {
         if (S_ISDIR(st.st_mode)) {
            // Directory
            Dir = Path;
         } else {
            // Regular file
            File = Path;
         }
      } else {
         if (errno == ENOENT) {
            // Path does not exist: assume this is the wanted file
            File = Path;
         } else {
            // Path exists but we cannot access it - exit
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            PRT("+ Cannot access requested path: "<<Path.c_str());
            PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            return 1;
         }
      }
   }

   // Default File, if not specified
   if (!File.length()) {
      if (!Dir.length())
         Dir = DirRef;
      // Expand File
      XrdSutExpand(Dir);
      File = Dir;
      // Make the directory, if needed
      if (XrdSutMkdir(File.c_str(),0777) != 0) {
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         PRT("+ Cannot create requested path: "<<File.c_str());
         PRT("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
         return 1;
      }
      // Define the files
      if (Mode == kM_admin) {
         File += AdminRef;
      } else if (Mode == kM_user) {
         File += UserRef;
      } else if (Mode == kM_netrc) {
         File += NetRcRef;
      } else if (Mode == kM_srvpuk) {
         File += SrvPukRef;
      }
   }

   return 0;
}

void ParseCrypto()
{
   // Parse crypto information in globals to load relevant factories

   // Use defaults if no special argument was entered
   if (CryptList == "")
      CryptList = DefCrypto;

   //
   // Vectorize
   int from = 0;
   while ((from = CryptList.tokenize(CryptMod[ncrypt], from, '|')) != -1
           && ncrypt < NCRYPTMAX) {
      ncrypt++;
   } 
   RefCip = new XrdCryptoCipher *[ncrypt];
   CF = new XrdCryptoFactory *[ncrypt];
   if (CF) {
      int i = 0;
      for (; i < ncrypt; i++ ) {
         // Get hook to crypto factory
         CF[i] = XrdCryptoFactory::GetCryptoFactory(CryptMod[i].c_str());
         if (!CF[i]) {
            PRT("// Hook for crypto factory "<<CryptMod[i]<<" undefined");
            continue;
         }
      }
   }
}

bool CheckOption(XrdOucString opt, const char *ref, int &ival)
{
   // Check opt against ref
   // Return 1 if ok, 0 if not
   // Fills ival = 1 if match is exact
   //       ival = 0 if match is exact with no<ref> 
   //       ival = -1 in the other cases
   bool rc = 0;

   int lref = (ref) ? strlen(ref) : 0;
   if (!lref) 
      return rc;
   XrdOucString noref = ref;
   noref.insert("no",0);

   ival = -1;
   if (opt == ref) {
      ival = 1;
      rc = 1;
   } else if (opt == noref) {
      ival = 0;
      rc = 1;
   }

   return rc;
}

bool AddPassword(XrdSutPFEntry &ent, XrdOucString salt, XrdOucString &ranpwd,
                 bool random, bool checkpw, bool &newpw)
{
   // Generate (prompting or randomly) new password and add it
   // to entry ent
   // If checkpw, make sure that it is different from the existing
   // one (check is done on the hash, cannot decide if the change
   // is significant or not).
   // Return generated random password in ranpwd.
   // Randoms passwords are 8 char lengths filled with upper and
   // lower case letters and numbers  
   // If !newpw, the a pwd saved during a previous call is used,
   // if any.
   // Return 1 if ok, 0 otherwise.
   static XrdOucString pwdref;

   XrdSutPFBuf oldsalt;
   XrdSutPFBuf oldhash;
   //
   // Save existing salt and hash, if required
   if (checkpw) {
      if (ent.buf1.len > 0 && ent.buf1.buf) {
         oldsalt.SetBuf(ent.buf1.buf,ent.buf1.len);
         if (ent.buf2.len > 0 && ent.buf2.buf) {
            oldhash.SetBuf(ent.buf2.buf,ent.buf2.len);
         } else {
            checkpw = 0;
         }
      } else {
         checkpw = 0;
      }
   }
   //
   // Save salt
   ent.buf1.SetBuf(salt.c_str(),salt.length());
   //
   // Prepare to get password
   XrdOucString passwd = "";
   if (newpw || !pwdref.length()) {
      newpw = 1;
      pwdref = "";
   }
   char *pwhash = 0;
   int pwhlen = 0;
   int natt = 0;
   while (!passwd.length()) {
      //
      //
      if (natt == kMAXPWDATT) {
         PRT("AddPassword: max number of attempts reached: "<<kMAXPWDATT);
         if (pwhash) delete[] pwhash;
         return 0;
      }
      //
      // Inquire password
      if (newpw) {
         if (!random) {
            XrdOucString prompt = "Password: ";
            if (natt == (kMAXPWDATT - 1))
               prompt.insert(" (last attempt)",prompt.find(":"));
            XrdSutGetPass(prompt.c_str(), passwd);
            if (passwd.length()) {
               pwdref = passwd;
               if (SavePw)
                  ranpwd = passwd;
               newpw = 0;
            } else {
               natt++;
               break;
            }
         } else if (random) {
            XrdSutRndm::GetString(1,8,passwd);
            if (IterNum.length() > 0) {
               // Set a non-default iteration number (we are going to hash
               // the password with itself)
               passwd.insert(IterNum,0);
            }
            pwdref = passwd;
            ranpwd = passwd;
            newpw = 0;
            checkpw = 0; // not needed
         }
      } else {
         passwd = pwdref;
      }
      // Get pw hash encoding password with itself
      pwhash = new char[(*KDFunLen)()];
      pwhlen = (*KDFun)(passwd.c_str(),passwd.length(),
                        passwd.c_str(),passwd.length(),pwhash,0);
      //
      // Check the password if required
      if (checkpw) {
         // Get hash with old salt
         char *osahash = new char[(*KDFunLen)()];
         // Encode the pw hash with the salt
         (*KDFun)(pwhash,pwhlen,
                  oldsalt.buf,oldsalt.len,osahash,0);
         if (!memcmp(oldhash.buf,osahash,oldhash.len)) {
            // Do not accept this password
            PRT("AddPassword: Password seems to be the same"
                  ": please enter a different one");
            passwd.hardreset();
            pwdref.hardreset();
            ranpwd.hardreset();
            newpw = 1;
         }
         // Cleanup
         if (osahash) delete[] osahash;
      }
   }
   //
   // Calculate new hash, now
   if (passwd.length()) {
      // Get new hash
      char *nsahash = new char[(*KDFunLen)()];
      // Encode first the hash with the salt
      int hlen = (*KDFun)(pwhash,pwhlen,
                          salt.c_str(),salt.length(),nsahash,0);
      // Copy result in buf 2
      ent.buf2.SetBuf(nsahash,hlen);
      // Cleanup
      if (nsahash) delete[] nsahash;
   }
   //
   // Cleanup
   if (pwhash) delete[] pwhash;
   // We are done
   return 1;
}

bool AddPassword(XrdSutPFEntry &ent, bool &newpw, const char *pwd)
{
   // Prompt new password and save in hash form to entry ent
   // (if pwd is defined, take password from pwd). 
   // If !newpw, the a pwd saved during a previous call is used,
   // if any.
   // Return 1 if ok, 0 otherwise.
   static XrdOucString pwdref;

   //
   // Prepare to get passwrod
   XrdOucString passwd = "";
   if (newpw || !pwdref.length()) {
      newpw = 1;
      pwdref = "";
   }
   //
   // If we are given a password, use it
   if (pwd && strlen(pwd) > 0) {
      PRT("AddPassword: using input password ("<<strlen(pwd)<<" bytes)");
      passwd = pwd;
   }
   char *pwhash = 0;
   int pwhlen = 0;
   int natt = 0;
   while (!passwd.length()) {
      //
      //
      if (natt == kMAXPWDATT) {
         PRT("AddPassword: max number of attempts reached: "<<kMAXPWDATT);
         if (pwhash) delete[] pwhash;
         return 0;
      }
      //
      // Inquire password
      if (newpw) {
         XrdOucString prompt = "Password: ";
         if (natt == (kMAXPWDATT - 1))
            prompt.insert(" (last attempt)",prompt.find(":"));
         XrdSutGetPass(prompt.c_str(), passwd);
         if (passwd.length()) {
            pwdref = passwd;
            newpw = 0;
         } else {
            natt++;
            break;
         }
      } else {
         passwd = pwdref;
      }
   }
   //
   // Get pw hash encoding password with itself
   if (Hash) {
      pwhash = new char[(*KDFunLen)()];
      pwhlen = (*KDFun)(passwd.c_str(),passwd.length(),
                        passwd.c_str(),passwd.length(),pwhash,0);
   } else {
      // Provided for backward compatibility with crypt-like
      // password hash: we just store the password in this case
      pwhlen = passwd.length();
      pwhash = new char[pwhlen];
      memcpy(pwhash,passwd.c_str(),pwhlen);
   }
   //
   // Save result in buf 1
   ent.buf1.SetBuf(pwhash,pwhlen);
   //
   // Cleanup
   if (pwhash) delete[] pwhash;
   // We are done
   return 1;
}

void SavePasswd(XrdOucString tag, XrdOucString pwd, bool onetime)
{
   // Save password pwd for tag in file
   
   // Make sure we gor something
   if (!tag.length() || !pwd.length()) {
      PRT("SavePasswd: tag or pwd undefined - do nothing ("<<
          tag.c_str()<<","<<pwd.c_str()<<")");
      return;
   }
   // Make sure the directory exists, first
   if (!Dir.length()) {
      PRT("SavePasswd: main directory undefined - do nothing");
      return;
   }
   //
   // Define passwd dir
   PwdFile = Dir;
   PwdFile += GenPwdRef;
   //
   // Make the directory, if needed
   if (XrdSutMkdir(PwdFile.c_str(),0777) != 0) {
      PRT("SavePasswd: Cannot create requested path: "<<PwdFile.c_str());
      return;
   }
   //
   // File name
   PwdFile += tag;
   //
   // Open file, truncating if it exists already
   int fd = open(PwdFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
   if (fd < 0) {
      PRT("SavePasswd: could not open/create file: "<<PwdFile.c_str());
      PRT("SavePasswd: errno: "<<errno);
      return;
   }
   //
   // Generate buffer
   XrdOucString buf;
   buf += "********* Password information **************\n\n";
   buf += "host:     "; buf += SrvName; buf += "\n";
   buf += "ID:       "; buf += SrvID; buf += "\n";
   buf += "tag:      "; buf += tag; buf += "\n";
   buf += "password: "; buf += pwd; buf += "\n";
   if (onetime) {
      buf += "status:   "; buf += 2; buf += "\n";
      buf += "\n";
      buf += "NB: one-time password: user will be asked for \n";
      buf += "    new password on first login               \n";
   } else {
      buf += "status:   "; buf += 1; buf += "\n";
      buf += "\n";
   }
   buf += "*********************************************";
   //
   // Write it to file
      // Now write the buffer to the stream
   while (write(fd, buf.c_str(), buf.length()) < 0 && errno == EINTR)
      errno = 0;
   //
   // Generate buffer
   buf.assign("\n",0);
   buf += "********* Server PUK information **************\n\n";
   int i = 0;
   for (; i < ncrypt; i++) {
      XrdOucString ptag = SrvName + ":";
      ptag += SrvID; ptag += "_"; ptag += CF[i]->ID();
      buf += "puk:      "; buf += ptag; buf += "\n";
      int lpub = 0;
      char *pub = RefCip[i]->Public(lpub);
      if (pub) {
         buf += pub; buf += "\n";
         delete[] pub;
      }
      buf += "epuk\n";
   }
   buf += "\n";
   buf += "*********************************************";
   //
   // Write it to file
      // Now write the buffer to the stream
   while (write(fd, buf.c_str(), buf.length()) < 0 && errno == EINTR)
      errno = 0;
   //
   // Close file
   close (fd);

   // We are done
   return;
}

bool GetEntry(XrdSutPFile *ff, XrdOucString tag,
              XrdSutPFEntry &ent, bool &check)
{
   // Get antry from file, checking force
   // Returns 1 if it exists and should not be updated
   // 0 otherwise

   int nr = ff->ReadEntry(tag.c_str(),ent);
   check = 0;
   if (nr > 0) {
      if (!Force) {
         PRT(" Entry for tag '"<<tag.c_str()<<
               "' already existing in file: "<<ff->Name());
         PRT(" Details: "<<ent.AsString());
         PRT(" Use option '-force' to overwrite / update");
         return 1;
      } else {
         check = 1;
      }
   } else {
      //
      // Prepare Entry
      ent.SetName(tag.c_str());
      ent.cnt    = 0;
   }
   return 0;
}

bool AskConfirm(const char *msg1, bool defact, const char *msg2)
{
   // Prompt for confirmation of action
   // If defined, msg1 is printed as prompt, followed by the default action
   // (  [y] == do-act, for defact = true; 
   //    [n] == do-not-act, for defact = false)
   // If defined, msg2 is printed before prompting.

   bool rc = defact;

   if (!Confirm) {
      rc = 1;
   } else {
      if (msg2) PRT(msg2);
      XrdOucString ask;
      XrdOucString prompt = defact ? " [y]: " : " [n]: ";
      if (msg1)
         prompt.insert(msg1,0);
      XrdSutGetLine(ask,prompt.c_str());
      ask.lower(0);
      if (ask.length()) {
         if (defact && (ask == 'n' || ask == "no")) {
            rc = 0;
         } else if (!defact && (ask == 'y' || ask == "yes")) {
            rc = 1;
         }
      }
   }
   // we are done
   return rc;
}

bool ReadPasswd(XrdOucString &tag, XrdOucString &pwd, int &st)
{
   // Read info from file PwdFile
   // Return tag in the form '<user>@<host><srvID>' and associated password
   
   // Make sure that the filename is defined
   if (PwdFile.length() <= 0) {
      PRT("ReadPasswd: file name undefined - do nothing");
      return 0;
   }
   //
   // Open file in read mode
   FILE *fd = fopen(PwdFile.c_str(),"r");
   if (fd == 0) {
      PRT("ReadPasswd: could not open file: "<<PwdFile.c_str());
      PRT("ReadPasswd: errno: "<<errno);
      return 0;
   }
   //
   // Read and process the info, now
   XrdOucString usr, host, id;
   char line[1024], s1[50], s2[1024];
   while (fgets(line, sizeof(line), fd) != 0) {
      if (line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = 0;
      if (strlen(line) <= 0)
         continue;
      if (sscanf(line,"%s %s",s1,s2) < 2)
         continue;
      if (!strncmp(s1,"host:",5)) {
         host = s2;
      } else if (!strncmp(s1,"ID:",3)) {
         id = s2;
      } else if (!strncmp(s1,"tag:",4)) {
         usr = s2;
      } else if (!strncmp(s1,"password:",9)) {
         pwd = s2;
      } else if (!strncmp(s1,"status:",7)) {
         st = strtol(s2, 0, 10);
      }
   }
   //
   // Close file
   fclose(fd);
   //
   // Check if we found all the essential information
   if (usr.length() <= 0 || pwd.length() <= 0) {
      if (usr.length() <= 0)
         PRT("ReadPasswd: usr tag missing in file "<<PwdFile.c_str());
      if (pwd.length() <= 0)
         PRT("ReadPasswd: password missing in file "<<PwdFile.c_str());
      return 0;
   }
   //
   // Warning if some other information is missing
   if (host.length() <= 0 || id.length() <= 0) {
      if (host.length() <= 0)
         PRT("ReadPasswd: warning: host name missing in file "
             <<PwdFile);
      if (id.length() <= 0)
         PRT("ReadPasswd: warning: srv ID missing in file "
             <<PwdFile);
   }
   //
   // Build tag
   tag = usr;
   //
   // Add host, if any
   if (host.length() > 0) {
      tag += '@';
      tag += host;
      tag += ':';
   }
   //
   // Add srv ID, if any
   if (id.length() > 0) {
      tag += id;
   }
   //
   // Notify tag
   PRT("ReadPasswd: build tag: "<<tag);


   // We are done
   return 1;
}

bool ReadPuk(int &ipuk, XrdOucString *tpuk, XrdOucString *puk)
{
   // Read server puks from file PwdFile
   // Return tags in the form '<host>:<srvID>_<cf_id>'
   
   // Make sure that the filename is defined
   if (PukFile.length() <= 0) {
      PRT("ReadPuk: file name undefined - do nothing");
      return 0;
   }
   //
   // Open file in read mode
   FILE *fd = fopen(PukFile.c_str(),"r");
   if (fd == 0) {
      PRT("ReadPuk: could not open file: "<<PukFile.c_str());
      PRT("ReadPuk: errno: "<<errno);
      return 0;
   }
   //
   // Read and process the info, now
   ipuk = 0;
   char line[1024], s1[50], s2[1024];
   while (fgets(line, sizeof(line), fd) != 0) {
      if (line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = 0;
      if (strlen(line) <= 0)
         continue;
      if (sscanf(line,"%s %s",s1,s2) < 2)
         continue;
      if (!strncmp(s1,"puk:",4)) {
         if (ipuk < kMAXPUK) {
            tpuk[ipuk] = s2;
            while (fgets(line, sizeof(line), fd) != 0) {
               if (!strncmp(line,"puk:",4) ||
                   !strncmp(line,"epuk",4) || strlen(line) <= 0)
                  break;
               puk[ipuk] += line;
            }
            ipuk++;
         } else {
            PRT("ReadPuk: warning: max number of puks reached ("<<kMAXPUK<<")");
         }
      }
   }
   //
   // Close file
   fclose(fd);
   //
   // Build puk tags
   PRT("ReadPuk: found "<<ipuk<<" server puks");
   int i = 0;
   for (; i < ipuk; i++) {
      //
      // Notify tag
      PRT("ReadPuk: build puk tag: "<<tpuk[i]);
   }

   // We are done
   return 1;
}

bool SavePuk()
{
   // Save ref ciphers in file named after GenPukRef and a date string

   // Make sure the directory exists, first
   if (!Dir.length()) {
      PRT("SavePuk: main directory undefined - do nothing");
      return 0;
   }
   //
   // Define passwd dir
   PukFile = Dir;
   PukFile += GenPukRef;
   //
   // Make the directory, if needed
   if (XrdSutMkdir(PukFile.c_str(),0777) != 0) {
      PRT("SavePuk: Cannot create requested path: "<<PukFile);
      return 0;
   }
   //
   // File name
   PukFile += "puk.";
   int now = time(0);
   char *tstr = new char[20];
   if (!tstr) {
      PRT("SavePuk: Cannot create buffer for time string");
      return 0;
   }
   XrdSutTimeString(now, tstr, 1);
   PukFile += tstr;
   delete [] tstr;
   //
   // Open file, truncating if it exists already
   int fd = open(PukFile.c_str(),O_WRONLY | O_CREAT | O_TRUNC, 0600);
   if (fd < 0) {
      PRT("SavePuk: could not open/create file: "<<PukFile);
      PRT("SavePuk: errno: "<<errno);
      return 0;
   }
   //
   // Temporary array of buckets
   XrdSutBucket **bck = new XrdSutBucket *[ncrypt];
   if (!bck) {
      PRT("SavePuk: Cannot create array of temporary buckets");
      return 0;
   }
   //
   // First loop over ciphers to determine the size
   int lout = 0, i = 0;
   for (; i < ncrypt; i++) {
      //
      // Make sure it is defined
      if (!CF[i] || !RefCip[i]) continue;
      //
      // Get bucket out of cipher
      bck[i] = RefCip[i]->AsBucket();
      if (!bck[i]) continue;
      //
      // Count
      lout += (bck[i]->size + 2*sizeof(kXR_int32));
   }
   //
   // Get the buffer
   char *bout = new char[lout];
   if (!bout) {
      PRT("SavePuk: Cannot create output buffer");
      close(fd);
      return 0;
   }
   //
   // Loop over ciphers to fill the buffer
   int lp = 0;
   for (i = 0; i < ncrypt; i++) {
      //
      // Make sure it is defined
      if (!CF[i] || !bck[i]) continue;
      //
      // The crypto ID first
      kXR_int32 id = CF[i]->ID();
      memcpy(bout+lp,&id,sizeof(kXR_int32));
      lp += sizeof(kXR_int32);
      //
      // The length second
      kXR_int32 lpuk = bck[i]->size;
      memcpy(bout+lp,&lpuk,sizeof(kXR_int32));
      lp += sizeof(kXR_int32);
      //
      // Finally the content
      memcpy(bout+lp,bck[i]->buffer,lpuk);
      lp += lpuk;
      //
      // Cleanup
      delete bck[i];
      bck[i] = 0;
   }
   delete[] bck;
   //
   // Write it to file
      // Now write the buffer to the stream
   while (write(fd, bout, lout) < 0 && errno == EINTR)
      errno = 0;
   PRT("SavePuk: "<<lout<<" bytes written to file "<<PukFile);
   //
   // Close file
   close (fd);

   // We are done
   return 1;
}

bool ReadPuk()
{
   // Read ref ciphers from file PukFile
   
   // Make sure that the filename is defined
   if (PukFile.length() <= 0) {
      PRT("ReadPuk: file name undefined - do nothing");
      return 0;
   }
   //
   // Open file in read mode
   int fd = open(PukFile.c_str(),O_RDONLY);
   if (fd < 0) {
      PRT("ReadPuk: could not open file: "<<PukFile.c_str());
      PRT("ReadPuk: errno: "<<errno);
      return 0;
   }

   //
   // Read out info now
   int nr = 0, nrdt = 0, ncip = 0;
   kXR_int32 id = 0, lpuk = 0;
   // the status ...
   while ((nr = read(fd,&id,sizeof(kXR_int32))) == sizeof(kXR_int32)) {
      nrdt += nr;
      // Read puk length
      if ((nr = read(fd,&lpuk,sizeof(kXR_int32))) != sizeof(kXR_int32)) {
         PRT("ReadPuk: could not read puk length - corrupton ? ");
         close(fd);
         return 0;
      }
      nrdt += nr;
      // Read puk buffer
      char *puk = new char[lpuk];
      if (!puk) {
         PRT("ReadPuk: could not allocate buffer for puk");
         close(fd);
         return 0;
      }
      if ((nr = read(fd, puk, lpuk)) != lpuk) {
         PRT("ReadPuk: could not read puk buffer - corrupton ? ");
         close(fd);
         return 0;
      }
      nrdt += nr;
      // Save in bucket
      XrdSutBucket *bck = new XrdSutBucket(puk, lpuk);
      if (!bck) {
         PRT("ReadPuk: could not create bucket for puk");
         delete[] puk;
         close(fd);
         return 0;
      }
      // Find crypto factory index
      int i = ncrypt - 1;
      while (i >= 0) {
         if (CF[i] && CF[i]->ID() == id) break;
         i--;
      }
      if (i < 0) {
         PRT("ReadPuk: warning: factory with ID "<< id << " not found");
         delete bck;
         continue;        
      }
      // Instantiate cipher from bucket
      RefCip[i] = CF[i]->Cipher(bck);     
      if (!RefCip[i]) {
         PRT("ReadPuk: warning: could not instantiate cipher"
             " from bucket for factory "<<CF[i]->Name());
      } else {
         PRT("ReadPuk: instantiate cipher for factory "<<CF[i]->Name());
      }
      // Count good ciphers
      ncip++;
      delete bck;
   }
   //
   // Close file
   close (fd);

   PRT("ReadPuk: "<<nrdt<<" bytes read from file "<<PukFile);
   PRT("ReadPuk: "<<ncip<<" ciphers instantiated");

   // We are done
   return 1;
}

int GeneratePuk()
{
   // Generate new ref ciphers for all the defined factories
   
   int ncf = 0, i = 0;
   for (; i < ncrypt; i++ ) {
      // Get hook to crypto factory
      CF[i] = XrdCryptoFactory::GetCryptoFactory(CryptMod[i].c_str());
      if (!CF[i]) {
         PRT("// Hook for crypto factory "<<CryptMod[i]<<" undefined");
         continue;
      }
      //
      // Generate reference cipher
      RefCip[i] = CF[i]->Cipher(0,0,0);
      if (!RefCip[i]) continue;
      //
      // Count success
      ncf++;
   }

   // We are done
   return ncf;
}

int LocateFactoryIndex(char *tag, int &id)
{
   // Searches tag for "_<id>" final strings
   // Extracts id and locate position in crypto array

   //
   // Locate factory ID
   XrdOucString sid(tag);
   sid.erase(0,sid.rfind('_')+1);
   id = atoi(sid.c_str());
   int j = ncrypt - 1;
   while (j >= 0) {
      if (CF[j] && CF[j]->ID() == id) break;
      j--;
   }
   if (j < 0)
      PRT("// warning: factory with ID "<< id << " not found");

   return j;
}

bool ExpPuk(const char *puk, bool read)
{
   // Export public part of key contained in file 'puk'. The file
   // name can be absolute or relative to the standard 'genpuk' or
   // a date to be looked for in the genpuk directory. The public
   // key is exported in a file adding the extension ".export"
   // to 'puk'. If the file name is not defined the most recent
   // key in the standard genpuk directory is exported.
   // Return 0 in case of failure, 1 in case of success.

   // Read the keys in, if needed
   if (read) {
      // Standard genpuk dir
      XrdOucString genpukdir = Dir;
      genpukdir += GenPukRef;

      // Locate the file with the full key
      if (puk && strlen(puk) > 0) {
         // If not absolute, expand with respect to the standard genpuk dir
         if (puk[0] != '/')
            PukFile = genpukdir;
         PukFile += puk;
      } else {
         // Scan the standard genpuk to find the most recent key
         DIR *dir = opendir(genpukdir.c_str());
         if (!dir) {
            PRT("ExpPuk: cannot open standard genpuk dir "<<genpukdir);
            return 0;
         }
         dirent *ent = 0;
         time_t latest = -1;
         while ((ent = readdir(dir))) {
            // Skip non-key files
            if (strncmp(ent->d_name, "puk.", 4))
               continue;
            // Get the modification date
            XrdOucString fn = genpukdir;
            fn += ent->d_name;
            struct stat st;
            if (stat(fn.c_str(), &st) != 0) {
               PRT("ExpPuk: cannot stat "<<fn<<" - skipping");
               continue;
            }
            if (st.st_mtime > latest) {
               PukFile = fn;
               latest = st.st_mtime;
            }
         }
      }

      // Read the keys in
      if (!ReadPuk()) {
         PRT("ExpPuk: problem reading the key in");
         return 0;
      }
   }

   // Build the export file name
   XrdOucString expfile = PukFile;
   expfile += ".export";
   PRT("ExpPuk: exporting key from file "<<PukFile);

   // Now we save the public part in the export files
   // Open file, truncating if it exists already
   int fd = open(expfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
   if (fd < 0) {
      PRT("ExpPuk: could not open/create file: "<<expfile.c_str());
      PRT("ExpPuk: errno: "<<errno);
      return 0;
   }
   //
   // Generate buffer
   XrdOucString buf;
   buf.assign("\n",0);
   buf += "********* Server PUK information **************\n\n";
   int i = 0;
   for (; i < ncrypt; i++) {
      XrdOucString ptag = SrvName + ":";
      ptag += SrvID; ptag += "_"; ptag += CF[i]->ID();
      buf += "puk:      "; buf += ptag; buf += "\n";
      int lpub = 0;
      char *pub = RefCip[i]->Public(lpub);
      if (pub) {
         buf += pub; buf += "\n";
         delete[] pub;
      }
      buf += "epuk\n";
   }
   buf += "\n";
   buf += "*********************************************";
   //
   // Write it to file
      // Now write the buffer to the stream
   while (write(fd, buf.c_str(), buf.length()) < 0 && errno == EINTR)
      errno = 0;
   //
   // Close file
   close (fd);

   // We are done
   return 1;
}
