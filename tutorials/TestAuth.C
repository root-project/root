///////////////////////////////////////////////////////////////////////////////
//
//  Macro test authentication methods stand alone
//
//  See $ROOTSYS/README/README.AUTH for additional details
//
//   Syntax:
//
//  .x TestAuth.C(<port>,"<user>","<krb5_princ>","<globus_det>")
//
//     <port>          = rootd port (default 1094)
//     <user>          = login user name for the test
//                       (default from getpwuid)
//     <krb5_princ>    = Principal to be used for Krb5 authentication
//                       in the form user@THE.REA.LM
//                       ( default: <running_user@Default_Realm with
//                                  Default_realm taken from /etc/krb5.conf
//                                  or the $KRB5_CONFIG file )
//     <globus_det>    = details for the globus authentication
//                       ( default: ad:certificates cd:$HOME/.globus
//                                  cf:usercert.pem kf:userkey.pem )
//
//  MAKE SURE that rootd is running
//
//  Example of successful output:
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                          +
// +                         TestAuth.C                                       +
// +                                                                          +
// +                Test of authentication methods                            +
// +                                                                          +
// +   Syntax:                                                                +
// +                                                                          +
// + .x TestAuth.C(<port>,"<user>","<krb5_princ>","<globus_det>")             +
// +                                                                          +
// +     <port>          = rootd port (default 1094)                          +
// +     <user>          = login user name for the test                       +
// +                      (default from getpwuid)                             +
// +     <krb5_princ>    = Principal to be used for Krb5 authentication       +
// +                       in the form user@THE.REA.LM                        +
// +                      ( default: <running_user@Default_Realm with         +
// +                                 Default_realm taken from /etc/krb5.conf  +
// +                                 or the $KRB5_CONFIG file )               +
// +     <globus_det>    = details for the globus authentication              +
// +                      ( default ad:certificates cd:$HOME/.globus          +
// +                                cf:usercert.pem kf:userkey.pem )          +
// +                                                                          +
// +                 >>> MAKE SURE that rootd is running <<<                  +
// +                                                                          +
// +             See $ROOTSYS/README/README.AUTH for additional details       +
// +                                                                          +
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                          +
// +   Basic test parameters:                                                 +
// +                                                                          +
// +   Local User is          : ganis
// +   Authentication Details : pt:0 ru:1 us:ganis
// +   Current directory is   : /home/ganis/local/root/root/tutorials
// +   TFTP string            : root://localhost:1094
// +   Krb5 Details           : pt:0 ru:1 us:ganis@PCEPSFT43.CERN.CH
// +                                                                          +
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                          +
// +   Testing UsrPwd ...                                                     +
// ganis@localhost password:
// +                                                                          +
// +   Testing SRP ...                                                        +
// ganis@localhost SRP password:
// +                                                                          +
// +   Testing Krb5 ...                                                       +
// Password for ganis@PCEPSFT43.CERN.CH:
// +                                                                          +
// +   Testing Globus ...                                                     +
// Local Globus Certificates (    )
// Enter <key>:<new value> to change:
// Your identity: /O=Grid/OU=GlobusTest/OU=simpleCA-arthux.cern.ch/OU=cern.ch/CN=ganis
// Enter GRID pass phrase for this identity:
// Creating proxy ............................ Done
// Your proxy is valid until: Fri Oct 31 09:33:04 2003
// +                                                                          +
// +   Testing SSH ...                                                        +
// ganis@localhost's password:
// +                                                                          +
// +   Testing UidGid ...                                                     +
// +                                                                          +
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                          +
// +   Result of the tests:                                                   +
// +                                                                          +
// +   Method: 0 (UsrPwd): successful! (reuse: successful!)                   +
// +   Method: 1    (SRP): successful! (reuse: successful!)                   +
// +   Method: 2   (Krb5): successful! (reuse: successful!)                   +
// +   Method: 3 (Globus): successful! (reuse: successful!)                   +
// +   Method: 4    (SSH): successful! (reuse: successful!)                   +
// +   Method: 5 (UidGid): successful!                                        +
// +                                                                          +
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
///////////////////////////////////////////////////////////////////////////////
//
int TestAuth(int port = 1094, char *user = "", char *krb5  = "", char *globus  = "")
{
   //
   // This macro tests the authentication methods
   //
   gROOT->Reset();

// Getting debug flag
   Int_t lDebug = gEnv->GetValue("Root.Debug",0);

// Useful flags
   Bool_t HaveMeth[6] = {1,0,0,0,0,1};
   Int_t  TestMeth[6] = {0,0,0,0,0,0};
   Int_t TestReUse[6] = {3,3,3,3,3,3};


// Some Printout
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("+                                                                             +\n");
   printf("+                         TestAuth.C                                          +\n");
   printf("+                                                                             +\n");
   printf("+                Test of authentication methods                               +\n");
   printf("+                                                                             +\n");
   printf("+   Syntax:                                                                   +\n");
   printf("+                                                                             +\n");
   printf("+ .x TestAuth.C(<port>,\"<user>\",\"<krb5_princ>\",\"<globus_det>\")                +\n");
   printf("+                                                                             +\n");
   printf("+     <port>          = rootd port (default 1094)                             +\n");
   printf("+     <user>          = login user name for the test                          +\n");
   printf("+                      (default from getpwuid)                                +\n");
   printf("+     <krb5_princ>    = Principal to be used for Krb5 authentication          +\n");
   printf("+                       in the form user@THE.REA.LM                           +\n");
   printf("+                      ( default: <running_user@Default_Realm with            +\n");
   printf("+                                 Default_realm taken from /etc/krb5.conf     +\n");
   printf("+                                 or the $KRB5_CONFIG file )                  +\n");
   printf("+     <globus_det>    = details for the globus authentication                 +\n");
   printf("+                      ( default ad:certificates cd:$HOME/.globus             +\n");
   printf("+                                cf:usercert.pem kf:userkey.pem )             +\n");
   printf("+                                                                             +\n");
   printf("+                     >>> MAKE SURE that rootd is running <<<                 +\n");
   printf("+                                                                             +\n");
   printf("+             See $ROOTSYS/README/README.AUTH for additional details          +\n");
   printf("+                                                                             +\n");
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

// Useful variables

// User
   TString User = user;
   if (User == "") {

      UserGroup_t *u = gSystem->GetUserInfo();
      if (!u) {
         printf("\n >>>> 'user' not defined: please enter a valid username:\n");
         char utmp[256] = {0};
         scanf("%s",utmp);
         if (strlen(utmp)) {
            User = utmp;
         } else {
            printf(">>>> no 'user' defined: return!\n");
            return 1;
         }
      } else {
         User = u->fUser;
      }

   }

// Host
   TString Host = "localhost";
   TString HostName = gSystem->HostName();

// File path string for TFTP
   //TString TFTPPath = TString("root://localhost:")+ port ;
   TString TFTPPath = TString("root://")+User+TString("@localhost:")+ port ;
   //TString TFTPPathKrb5 = TString("root://") + HostName + TString(":")+ port ;
   TString TFTPPathKrb5 = TString("root://") + User+ TString("@") +
                          HostName + TString(":")+ port ;

// Details
   TString Details = TString("pt:0 ru:1 us:") + User;

// Testing availibilities
   char *p;

//   TString HaveSRP = "@srpdir@";
   if ((p = gSystem->DynamicPathName("libSRPAuth", kTRUE))) {
      HaveMeth[1] = 1;
   }
   delete[] p;

// Check if Kerberos is available
   TString Krb5Details;
   TString Krb5Open;
   if ((p = gSystem->DynamicPathName("libKrb5Auth", kTRUE))) {
      HaveMeth[2] = 1;
      // Special details string for Kerberos
      if (strlen(krb5) > 0) {
         Krb5Details = TString("pt:0 ru:1 us:") + TString(krb5);
      } else {
         // Must determine a default ... look in config file
         TString Krb5Conf, Realm;
         if (gSystem->Getenv("KRB5_CONFIG")) {
            if (!gSystem->AccessPathName(gSystem->Getenv("KRB5_CONFIG"), kReadPermission)) {
               Krb5Conf = gSystem->Getenv("KRB5_CONFIG");
            }
         } else if (!gSystem->AccessPathName("/etc/krb5.conf", kReadPermission)) {
            Krb5Conf = "/etc/krb5.conf";
         } else {
            printf("\n >>>> Kerberos Principal undefined\n");
            printf("\n >>>> unable to localize Kerberos config file to build a default\n");
            printf("\n >>>> Switching off Kerberos\n");
            printf("\n >>>> Run again with giving the principal as 3rd argument\n");
            printf("\n >>>> or define the variable KRB5_CONFIG with the full path \n");
            printf("\n >>>> to the config file (usually /etc/krb5.conf)\n");
            HaveMeth[2] = 0;
         }
         if (HaveMeth[2] == 1) {
            FILE *fc = fopen(Krb5Conf.Data(),"r");
            if (fc) {
               char line[1024], fs1[1024], fs2[1024], fs3[1024];
               while (fgets(line, sizeof(line), fc) != 0) {
                  int nf = sscanf(line,"%s %s %s",fs1,fs2,fs3);
                  if (nf == 3 && !strcmp(fs1,"default_realm")) {
                     Realm = fs3;
                     break;
                  }
               }
               Krb5Details = TString("pt:0 ru:1 us:") + User + TString("@") + Realm;
//               Krb5Details = TString("pt:0 ru:1 us:") + User;
               fclose(fc);
            } else {
               HaveMeth[2] = 0;
            }
         }
      }
   }
   delete[] p;

// Check if Globus is available
   TString GlobusDetails;
   if ((p = gSystem->DynamicPathName("libGlobusAuth", kTRUE))) {
      HaveMeth[3] = 1;
      // Special details string for Globus
      GlobusDetails = TString("pt:0 ru:1 ") + TString(globus);
   }
   delete[] p;

// Check if SSH available
   if (gSystem->Which(gSystem->Getenv("PATH"), "ssh", kExecutePermission)) {
      HaveMeth[4] = 1;
   }

// Test parameter Printout
   printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("+                                                                             +\n");
   printf("+   Basic test parameters:                                                    +\n");
   printf("+                                                                             +\n");
   printf("+   Local User is          : %s \n",User.Data());
   printf("+   Authentication Details : %s \n",Details.Data());
   printf("+   Current directory is   : %s \n",gSystem->WorkingDirectory());
   printf("+   TFTP string            : %s \n",TFTPPath.Data());
   if (HaveMeth[2]) {
      printf("+   Krb5 Details           : %s \n",Krb5Details.Data());
   }
   printf("+                                                                             +\n");
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

// Read local <RootAuthrc> now to avoid to be later superseeded
   TAuthenticate::ReadRootAuthrc();
   if (lDebug > 0)
      TAuthenticate::Show();

   TFTP *t1 = 0;
// UsrPwd method
   printf("+                                                                             +\n");
   printf("+   Testing UsrPwd ...                                                        +\n");

   // Check if by any chance locally there is already an THostAuth matching
   // the one we want to use for testing
   THostAuth *hasv1 = 0;
   THostAuth *ha = TAuthenticate::HasHostAuth(Host.Data(),User.Data());
   if (ha) {
      // We need to save it to restore at the end
      hasv1 = new THostAuth(*ha);
      // We reset the existing one
      ha->Reset();
      // And update it with the info we want
      ha->AddMethod(0,Details.Data());
   } else {
      // We create directly a new THostAuth
      ha = new THostAuth(Host.Data(),User.Data(),0,Details.Data());
      // And add object to list so that TAuthenticate has
      // a chance to find it
      TAuthenticate::GetAuthInfo()->Add(ha);
   }

   // Print available host auth info
   if (lDebug > 0)
      ha->Print();

   {
   // First authentication attempt
   t1 = new TFTP(TFTPPath.Data(),2);
   if (t1->IsOpen()) {
      TestMeth[0] = 1;
   } else {
      printf(" >>>>>>>>>>>>>>>> Test of UsrPwd authentication failed \n");
   }}

   // Try ReUse
   if (TestMeth[0] == 1) {
      TIter next(ha->Established());
      TSecContext *ai;
      while ((ai = (TSecContext *) next())) {
         if (ai->GetMethod() == 0) {
            Int_t OffSet = ai->GetOffSet();
            TestReUse[0] = 0;
            if (OffSet > -1) {
               TestReUse[0] = 1;
            }
         }
      }
   }
   // Delete t1
   if (t1) delete t1;
   // remove method from available list
   ha->RemoveMethod(0);

// SRP method
   if ( HaveMeth[1] ) {
      printf("+                                                                             +\n");
      printf("+   Testing SRP ...                                                           +\n");

     // Add relevant info to HostAuth
      ha->AddFirst(1,Details.Data());
      if (lDebug > 0)
         ha->Print();

     // Authentication attempt
      t1 = new TFTP(TFTPPath.Data(),2);
      if (t1->IsOpen()) {
         TestMeth[1] = 1;
      } else {
         printf(" >>>>>>>>>>>>>>>> Test of SRP authentication failed \n");
      }

      // Try ReUse
      if (TestMeth[1] == 1) {
         TIter next(ha->Established());
         TSecContext *ai;
         while ((ai = (TSecContext *) next())) {
            if (ai->GetMethod() == 1) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[1] = 0;
               if (OffSet > -1) {
                  TestReUse[1] = 1;
               }
            }
         }
      }
      // Delete t1
      if (t1) delete t1;
      // remove method from available list
      ha->RemoveMethod(1);

   }

// Kerberos method
   THostAuth *hak = 0;
   THostAuth *hasv2 = 0;
   if ( HaveMeth[2] ) {
      printf("+                                                                             +\n");
      printf("+   Testing Krb5 ...                                                          +\n");


      // Check if by any chance locally there is already an THostAuth matching
      // the one we want to use for testing
      hak = TAuthenticate::HasHostAuth(HostName.Data(),User.Data());
      if (hak) {
         if (lDebug > 0)
            printf(" >>>>>>>>>>>>>>>> Krb5: existing THostAuth found \n");
         // We need to save it to restore at the end
         hasv2 = new THostAuth(*hak);
         // We reset the existing one
         hak->Reset();
         // And update it with the info we want
         hak->AddMethod(2,Krb5Details.Data());
      } else {
         // We create directly a new THostAuth
         hak = new THostAuth(HostName.Data(),User.Data(),2,Krb5Details.Data());
         // And add object to list so that TAuthenticate has
         // a chance to find it
         TAuthenticate::GetAuthInfo()->Add(hak);
      }
      if (lDebug > 0)
         hak->Print();

     // Authentication attempt
      t1 = new TFTP(TFTPPathKrb5.Data(),2);
      if (t1->IsOpen()) {
         TestMeth[2] = 1;
      } else {
         printf(" >>>>>>>>>>>>>>>> Test of Kerberos authentication failed \n");
         if (strlen(krb5) > 0) {
            printf(" >>>>>>>>>>>>>>>> details used: '%s' \n",krb5);
         }
      }

      // Try ReUse
      if (TestMeth[2] == 1) {
         TIter next(hak->Established());
         TSecContext *ai;
         while ((ai = (TSecContext *) next())) {
            if (ai->GetMethod() == 2) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[2] = 0;
               if (OffSet > -1) {
                  TestReUse[2] = 1;
               }
            }
         }
      }
      // Delete t1
      if (t1) delete t1;
      // remove method from available list
      hak->RemoveMethod(2);
   }

// Globus method
   if ( HaveMeth[3] ) {
      printf("+                                                                             +\n");
      printf("+   Testing Globus ...                                                        +\n");

     // Add relevant info to HostAuth
      ha->AddFirst(3,GlobusDetails.Data());
      if (lDebug > 0)
         ha->Print();

     // Authentication attempt
      t1 = new TFTP(TFTPPath.Data(),2);
      if (t1->IsOpen()) {
         TestMeth[3] = 1;
      } else {
         printf(" >>>>>>>>>>>>>>>> Test of Globus authentication failed \n");
         if (strlen(globus) > 0) {
            printf(" >>>>>>>>>>>>>>>> details used: '%s' \n",globus);
         } else {
            printf(" >>>>>>>>>>>>>>>> using default details: \n");
            printf(" >>>>>>>>>>>>>>>>   ad:/etc/grid-security/certificates");
            printf(" cd:$HOME/.globus cf:usercert.pem kf:userkey.pem\n");
         }
         UserGroup_t *u = gSystem->GetUserInfo();
         if (u) {
            if (u->fUid > 0) {
               printf(" >>>>>>>>>>>>>>>> You are not root,");
               printf(" you may not have the right privileges\n");
               printf(" >>>>>>>>>>>>>>>> Make sure that the used details are correct! \n");
            }
         }
      }

      // Try ReUse
      if (TestMeth[3] == 1) {
         TIter next(ha->Established());
         TSecContext *ai;
         while ((ai = (TSecContext *) next())) {
            if (ai->GetMethod() == 3) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[3] = 0;
               if (OffSet > -1) {
                  TestReUse[3] = 1;
               }
            }
         }
      }
      // Delete t1
      if (t1) delete t1;
      // remove method from available list
      ha->RemoveMethod(3);
   }

// SSH method

   if ( HaveMeth[4] ) {
      printf("+                                                                             +\n");
      printf("+   Testing SSH ...                                                           +\n");

     // Add relevant info to HostAuth
      ha->AddFirst(4,Details.Data());
      if (lDebug > 0)
         ha->Print();

     // Authentication attempt
      t1 = new TFTP(TFTPPath.Data(),2);
      if (t1->IsOpen()) {
         TestMeth[4] = 1;
      } else {
         printf(" >>>>>>>>>>>>>>>> Test of SSH authentication failed \n");
      }

      // Try ReUse
      if (TestMeth[4] == 1) {
         TIter next(ha->Established());
         TSecContext *ai;
         while ((ai = (TSecContext *) next())) {
            if (ai->GetMethod() == 4) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[4] = 0;
               if (OffSet > -1) {
                  TestReUse[4] = 1;
               }
            }
         }
      }
      // Delete t1
      if (t1) delete t1;
      // remove method from available list
      ha->RemoveMethod(4);
   }


// Rfio method
   printf("+                                                                             +\n");
   printf("+   Testing UidGid ...                                                        +\n");

   // Add relevant info to HostAuth
   ha->AddFirst(5,Details.Data());
   if (lDebug > 0)
      ha->Print();

   // Authentication attempt
   {
   t1 = new TFTP(TFTPPath.Data(),2);
   if (t1->IsOpen()) {
      TestMeth[5] = 1;
   } else {
      printf(" >>>>>>>>>>>>>>>> Test of UidGid authentication failed \n");
   }}

   // Delete t1
   if (t1) delete t1;

   // remove method from available list
   ha->RemoveMethod(5);

   printf("+                                                                             +\n");
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

   // Print available host auth info
   if (lDebug > 0)
      TAuthenticate::Show();

// Now restore initial configuration
   if (hasv1) {
      ha->Reset();
      ha->Update(hasv1);
   } else {
      TAuthenticate::GetAuthInfo()->Remove(ha);
   }
   if (hasv2) {
      hak->Reset();
      hak->Update(hasv2);
   } else {
      TAuthenticate::GetAuthInfo()->Remove(hak);
   }

// Final Printout
   printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("+                                                                             +\n");
   printf("+   Result of the tests:                                                      +\n");
   printf("+                                                                             +\n");
   char status[4][20] = {"failed!","successful!","not testable","not tested"};
   int i = 0;
   for( i=0; i<6; i++ ) {
     if (HaveMeth[i] && TestMeth[i] < 2) {
       if (i < 5) {
          printf("+   Method: %d %8s: %11s (reuse: %11s)                      +\n",i,
                         Form("(%s)",TAuthenticate::GetAuthMethod(i)),
                         status[TestMeth[i]],status[TestReUse[i]]);
       } else
          printf("+   Method: %d %8s: %11s                                           +\n",i,
                         Form("(%s)",TAuthenticate::GetAuthMethod(i)),
                         status[TestMeth[i]]);
     }
   }
   Bool_t NotPrinted = kTRUE;
   for( i=0; i<6; i++ ) {
     if (HaveMeth[i] && TestMeth[i] > 1) {
        if (NotPrinted) {
           printf("+                                                                             +\n");
           printf("+   Could not be tested:                                                      +\n");
           printf("+                                                                             +\n");
           NotPrinted = kFALSE;
        }
        printf("+   Method: %d %8s: %11s                      +\n",i,
                       Form("(%s)",TAuthenticate::GetAuthMethod(i)),
                       status[TestMeth[i]]);
     }
   }
   printf("+                                                                             +\n");
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

}
