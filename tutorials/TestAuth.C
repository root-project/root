//////////////////////////////////////////////////////////////////////
//
//  Macro test authentication methods stand alone
//
//  NB: Kerberos cannot be tested in standalone
//
//
//   Syntax:
//
//  .x TestAuth.C("<user>","<write path>",<port>,"<globus_det>")
//
//     <user>          = login user name for the test
//                      (default from getpwuid)
//     <write path>    = path writable by <user>
//                      (default /tmp)
//     <port>          = rootd port (default 1094)
//     <globus_det>    = details for the globus authentication
//                       ( default: ad:certificates cd:$HOME/.globus
//                                  cf:usercert.pem kf:userkey.pem )
//
//  MAKE SURE that rootd is running
//
//  Example of successful output:
//
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//    +                                                                   +
//    +                         TestAuth.C                                +
//    +                                                                   +
//    +                Test of authentication methods                     +
//    +                                                                   +
//    +   Syntax:                                                         +
//    +                                                                   +
//    + .x TestAuth.C("<user>","<write path>",<port>,"<globus_det>")      +
//    +                                                                   +
//    +     <user>          = login user name for the test                +
//    +                      (default from getpwuid)                      +
//    +     <write path>    = path writable by <user>                     +
//    +                      (default /tmp)                               +
//    +     <port>          = rootd port (default 1094)                   +
//    +     <globus_det>    = details for the globus authentication       +
//    +                      ( default ad:certificates cd:$HOME/.globus   +
//    +                                cf:usercert.pem kf:userkey.pem )   +
//    +                                                                   +
//    +           >>> MAKE SURE that rootd is running <<<                 +
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//    +                                                         +
//    +   Basic test parameters:                                +
//    +                                                         +
//    +   Local User is          : ganis
//    +   Authentication Details : pt:0 ru:1 us:ganis
//    +   Current directory is   : /afs/cern.ch/aleph/scratch/ganis/root/Linux.Standard/tutorials
//    +   Test File              : /tmp/TestFile.root
//    +   TFile::Open string     : root://localhost:5000//tmp/TestFile.root
//    +                                                         +
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//    +                                                         +
//    +   Testing UsrPwd ...                                    +
// qwerty@localhost password:
//    +                                                         +
//    +   Testing SRP ...                                       +
// qwerty@localhost SRP password:
//    +                                                         +
//    +   Testing Krb5 ...                                      +
//    +   Krb5 authentication cannot be tested in standalone    +
//    +                                                         +
//    +   Testing Globus ...                                    +
// Local Globus Certificates (    )
// Enter <key>:<new value> to change:
// Your identity: /O=Grid/OU=GlobusTest/OU=simpleCA-arthux.cern.ch/OU=cern.ch/CN=ganis
// Enter GRID pass phrase for this identity:
// Creating proxy ............................ Done
// Your proxy is valid until: Sat Oct 18 14:23:00 2003
//    +                                                         +
//    +   Testing SSH ...                                       +
// qwerty@localhost's password:
//    +                                                         +
//    +   Testing UidGid ...                                    +
//    +                                                         +
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//    +                                                         +
//    +   Result of the tests:                                  +
//    +                                                         +
//    +   Method: 0 (UsrPwd): successful! (reuse: successful!)  +
//    +   Method: 1    (SRP): successful! (reuse: successful!)  +
//    +   Method: 3 (Globus): successful! (reuse: successful!)  +
//    +   Method: 4    (SSH): successful! (reuse: successful!)  +
//    +   Method: 5 (UidGid): successful!                       +
//    +                                                         +
//    +   Could not be tested:                                  +
//    +                                                         +
//    +   Method: 2   (Krb5): not testable                      +
//    +                                                         +
//    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
///////////////////////////////////////////////////////////////////////
//
int TestAuth(char *user = "", char *workdir = "/tmp",
             int port = 1094, char *globus  = "")
{
   //
   // This macro tests the authentication methods
   //
   gROOT->Reset();

// Getting debug flag
   Int_t lDebug = gEnv->GetValue("Root.Debug",0);

// Setting test flag
   gEnv->SetValue("Test.Auth",1);

// Useful flags
   Bool_t HaveMeth[6] = {1,0,0,0,0,1};
   Int_t  TestMeth[6] = {0,0,0,0,0,0};
   Int_t TestReUse[6] = {3,3,3,3,3,3};

// Testing availibilities
   char *p;

//   TString HaveSRP = "@srpdir@";
   if ((p = gSystem->DynamicPathName("libSRPAuth", kTRUE))) {
      HaveMeth[1] = 1;
   }
   delete[] p;

// Check if Kerberos is available
   if ((p = gSystem->DynamicPathName("libKrb5Auth", kTRUE))) {
      HaveMeth[2] = 1;
   }
   delete[] p;

// Check if Globus is available
   if ((p = gSystem->DynamicPathName("libGlobusAuth", kTRUE))) {
      HaveMeth[3] = 1;
   }
   delete[] p;

// Check if SSH available
   if (gSystem->Which(gSystem->Getenv("PATH"), "ssh", kExecutePermission)) {
      HaveMeth[4] = 1;
   }

// Some Printout
   printf("   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("   +                                                                   +\n");
   printf("   +                         TestAuth.C                                +\n");
   printf("   +                                                                   +\n");
   printf("   +                Test of authentication methods                     +\n");
   printf("   +                                                                   +\n");
   printf("   +   Syntax:                                                         +\n");
   printf("   +                                                                   +\n");
   printf("   + .x TestAuth.C(\"<user>\",\"<write path>\",<port>,\"<globus_det>\")      +\n");
   printf("   +                                                                   +\n");
   printf("   +     <user>          = login user name for the test                +\n");
   printf("   +                      (default from getpwuid)                      +\n");
   printf("   +     <write path>    = path writable by <user>                     +\n");
   printf("   +                      (default /tmp)                               +\n");
   printf("   +     <port>          = rootd port (default 1094)                   +\n");
   printf("   +     <globus_det>    = details for the globus authentication       +\n");
   printf("   +                      ( default ad:certificates cd:$HOME/.globus   +\n");
   printf("   +                                cf:usercert.pem kf:userkey.pem )   +\n");
   printf("   +                                                                   +\n");
   printf("   +           >>> MAKE SURE that rootd is running <<<                 +\n");
   printf("   +                                                                   +\n");
   printf("   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

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

// Working Dir
   TString WorkDir = workdir;
   if (WorkDir == "" ||
       gSystem->AccessPathName(WorkDir, kWritePermission)) {
       if (lDebug > 0)
          printf(">>>> Path: %s not writable!\n",WorkDir.Data());
       WorkDir = "/tmp";
       if (gSystem->AccessPathName(WorkDir, kWritePermission)) {
          if (lDebug > 0)
             printf(">>>> Path: %s not writable!\n",WorkDir.Data());
          WorkDir = gSystem->WorkingDirectory();
          if (WorkDir == "" ||
             gSystem->AccessPathName(WorkDir, kWritePermission)) {
             if (lDebug > 0)
                printf(">>>> Path: %s not writable!\n",WorkDir.Data());
             printf("unable to find writable dir for temporary test file: return!\n");
             return 1;
	  }
       }
   }

// TempFile path
   TString FileTemp = WorkDir + "/TestFile.root";

// Create test file first
   TFile *TestFile = new TFile(FileTemp.Data(),"RECREATE");
   if (TestFile && TestFile->IsOpen()) {
     Real_t theVector[5] = {0.,1.,2.,3.,4.};
     TVector vc(5,theVector);
     vc.Write();
     if (lDebug > 1)
        TestFile->ls();
     if (TestFile) delete TestFile;
   } else {
     printf("Cannot open test file: return!\n");
     return 1;
   }

// File path string for TFile::Open
   TString WkgFile =
      TString("root://localhost:") + port + "/" + WorkDir + "/TestFile.root";

// Host
   TString Host = "localhost";

// Details
   TString Details = TString("pt:0 ru:1 us:") + User;
   TString GlobusDetails = TString("pt:0 ru:1 ") + TString(globus);

// Test parameter Printout
   printf("\n   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("   +                                                         +\n");
   printf("   +   Basic test parameters:                                +\n");
   printf("   +                                                         +\n");
   printf("   +   Local User is          : %s \n",User.Data());
   printf("   +   Authentication Details : %s \n",Details.Data());
   printf("   +   Current directory is   : %s \n",gSystem->WorkingDirectory());
   printf("   +   Test File              : %s \n",FileTemp.Data());
   printf("   +   TFile::Open string     : %s \n",WkgFile.Data());
   printf("   +                                                         +\n");
   printf("   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");


// UsrPwd method
   printf("   +                                                         +\n");
   printf("   +   Testing UsrPwd ...                                    +\n");

   // Create a THostAuth instantiation for the local host
   THostAuth *hawk = new THostAuth(Host.Data(),User.Data(),0,Details.Data());

   // Add object to list
   TAuthenticate::GetAuthInfo()->Add(hawk);

   // Print available host auth info
   if (lDebug > 0)
      hawk->Print();

   // First authentication attempt
   TFile *f1 = TFile::Open(WkgFile.Data(),"read");
   if (f1 && f1->IsOpen()) {
      TestMeth[0] = 1;
      if (lDebug > 1)
         f1->ls();
   } else {
      printf(" >>>>>>>>>>>>>>>> Test of UsrPwd authentication failed \n");
   }
   // Delete file
   if (f1) delete f1;

   // Get pointer to relevant HostAuth
   THostAuth *ha = TAuthenticate::GetHostAuth(Host.Data(),User.Data());

   // Try ReUse
   if (TestMeth[0] == 1) {
      TIter next(ha->Established());
      TAuthDetails *ai;
      while ((ai = (TAuthDetails *) next())) {
         if (ai->GetMethod() == 0) {
            Int_t OffSet = ai->GetOffSet();
            TestReUse[0] = 0;
            if (OffSet > -1) {
               TestReUse[0] = 1;
            }
         }
      }
   }
   // remove method from available list
   ha->RemoveMethod(0);

// SRP method
   if ( HaveMeth[1] ) {
      printf("   +                                                         +\n");
      printf("   +   Testing SRP ...                                       +\n");

     // Add relevant info to HostAuth
      ha->SetFirst(1,Details.Data());
      if (lDebug > 0)
         ha->Print();

     // Authentication attempt
      TFile *f1 = TFile::Open(WkgFile.Data(),"read");
      if (f1 && f1->IsOpen()) {
         TestMeth[1] = 1;
         if (lDebug > 1)
            f1->ls();
      } else {
         printf(" >>>>>>>>>>>>>>>> Test of SRP authentication failed \n");
      }
       // Delete file
      if (f1) delete f1;

      // Try ReUse
      if (TestMeth[1] == 1) {
         TIter next(ha->Established());
         TAuthDetails *ai;
         while ((ai = (TAuthDetails *) next())) {
            if (ai->GetMethod() == 1) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[1] = 0;
               if (OffSet > -1) {
                  TestReUse[1] = 1;
               }
            }
         }
      }
      // remove method from available list
      ha->RemoveMethod(1);

   }

// Kerberos method
   if ( HaveMeth[2] ) {
      printf("   +                                                         +\n");
      printf("   +   Testing Krb5 ...                                      +\n");

      // Standalone test not possible
      TestMeth[2] = 2;
      printf("   +   Krb5 authentication cannot be tested in standalone    +\n");
   }


// Globus method
   if ( HaveMeth[3] ) {
      printf("   +                                                         +\n");
      printf("   +   Testing Globus ...                                    +\n");

     // Add relevant info to HostAuth
      ha->SetFirst(3,GlobusDetails.Data());
      if (lDebug > 0)
         ha->Print();

     // Authentication attempt
      TFile *f1 = TFile::Open(WkgFile.Data(),"read");
      if (f1 && f1->IsOpen()) {
         TestMeth[3] = 1;
         if (lDebug > 1)
            f1->ls();
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
       // Delete file
      if (f1) delete f1;

      // Try ReUse
      if (TestMeth[3] == 1) {
         TIter next(ha->Established());
         TAuthDetails *ai;
         while ((ai = (TAuthDetails *) next())) {
            if (ai->GetMethod() == 1) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[3] = 0;
               if (OffSet > -1) {
                  TestReUse[3] = 1;
               }
            }
         }
      }
      // remove method from available list
      ha->RemoveMethod(3);
   }

// SSH methodg

   if ( HaveMeth[4] ) {
      printf("   +                                                         +\n");
      printf("   +   Testing SSH ...                                       +\n");

     // Add relevant info to HostAuth
      ha->SetFirst(4,Details.Data());
      if (lDebug > 0)
         ha->Print();

     // Authentication attempt
      TFile *f1 = TFile::Open(WkgFile.Data(),"read");
      if (f1 && f1->IsOpen()) {
         TestMeth[4] = 1;
         if (lDebug > 0)
            f1->ls();
      } else {
         printf(" >>>>>>>>>>>>>>>> Test of SSH authentication failed \n");
      }
       // Delete file
      if (f1) delete f1;

      // Try ReUse
      if (TestMeth[4] == 1) {
         TIter next(ha->Established());
         TAuthDetails *ai;
         while ((ai = (TAuthDetails *) next())) {
            if (ai->GetMethod() == 4) {
               Int_t OffSet = ai->GetOffSet();
               TestReUse[4] = 0;
               if (OffSet > -1) {
                  TestReUse[4] = 1;
               }
            }
         }
      }
      // remove method from available list
      ha->RemoveMethod(4);
   }


// Rfio method
   printf("   +                                                         +\n");
   printf("   +   Testing UidGid ...                                    +\n");

   // Add relevant info to HostAuth
   ha->SetFirst(5,Details.Data());
   if (lDebug > 0)
      ha->Print();

   // Authentication attempt
   TFile *f1 = TFile::Open(WkgFile.Data(),"read");
   if (f1 && f1->IsOpen()) {
      TestMeth[5] = 1;
      if (lDebug > 1)
         f1->ls();
   } else {
      printf(" >>>>>>>>>>>>>>>> Test of UidGid authentication failed \n");
   }
   // Delete file
   if (f1) delete f1;

   // remove method from available list
   ha->RemoveMethod(5);

   printf("   +                                                         +\n");
   printf("   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

   // Print available host auth info
   if (lDebug > 0)
      TAuthenticate::PrintHostAuth();

// Delete file
   gSystem->Unlink("TestFile.root");

// Setting off test flag
   gEnv->SetValue("Test.Auth",0);

// Final Printout
   printf("\n   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("   +                                                         +\n");
   printf("   +   Result of the tests:                                  +\n");
   printf("   +                                                         +\n");
   char status[4][20] = {"failed!","successful!","not testable","not tested"};
   int i = 0;
   for( i=0; i<6; i++ ) {
     if (HaveMeth[i] && TestMeth[i] < 2) {
       if (i < 5) {
          printf("   +   Method: %d %8s: %11s (reuse: %11s)  +\n",i,
                         Form("(%s)",TAuthenticate::GetAuthMethod(i)),
                         status[TestMeth[i]],status[TestReUse[i]]);
       } else
          printf("   +   Method: %d %8s: %11s                       +\n",i,
                         Form("(%s)",TAuthenticate::GetAuthMethod(i)),
                         status[TestMeth[i]]);
     }
   }
   printf("   +                                                         +\n");
   printf("   +   Could not be tested:                                  +\n");
   printf("   +                                                         +\n");
   for( i=0; i<6; i++ ) {
     if (HaveMeth[i] && TestMeth[i] > 1) {
	printf("   +   Method: %d %8s: %11s                      +\n",i,
                       Form("(%s)",TAuthenticate::GetAuthMethod(i)),
                       status[TestMeth[i]]);
     }
   }
   printf("   +                                                         +\n");
   printf("   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

}
