/// \file
/// \ingroup tutorial_net
///  Macro test authentication methods stand alone
///
///  See `$ROOTSYS/README/README.AUTH` for additional details
///
///   Syntax:
///
/// ~~~ {.cpp}
///  .x TestAuth.C(<port>,"<user>")
///
///     <port>          = rootd port (default 1094)
///     <user>          = login user name for the test
///                       (default from getpwuid)
/// ~~~
///
///  MAKE SURE that rootd is running
///
///  Example of successful output:
///
/// ~~~ {.cpp}
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +                                                                          +
/// +                         TestAuth.C                                       +
/// +                                                                          +
/// +                Test of authentication methods                            +
/// +                                                                          +
/// +   Syntax:                                                                +
/// +                                                                          +
/// + .x TestAuth.C(<port>,"<user>")                                           +
/// +                                                                          +
/// +     <port>          = rootd port (default 1094)                          +
/// +     <user>          = login user name for the test                       +
/// +                      (default from getpwuid)                             +
/// +                                                                          +
/// +                 >>> MAKE SURE that rootd is running <<<                  +
/// +                                                                          +
/// +             See $ROOTSYS/README/README.AUTH for additional details       +
/// +                                                                          +
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +                                                                          +
/// +   Basic test parameters:                                                 +
/// +                                                                          +
/// +   Local User is          : ganis
/// +   Authentication Details : pt:0 ru:1 us:ganis
/// +   Current directory is   : /home/ganis/local/root/root/tutorials
/// +   TFTP string            : root://localhost:1094
/// +                                                                          +
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +                                                                          +
/// +   Testing UsrPwd ...                                                     +
/// ganis@localhost password:
/// +                                                                          +
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// +                                                                          +
/// +   Result of the tests:                                                   +
/// +                                                                          +
/// +   Method: 0 (UsrPwd): successful! (reuse: successful!)                   +
/// +                                                                          +
/// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/// ~~~
///
/// \macro_code
///
/// \author

int TestAuth(int port = 1094, char *user = "")
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
   printf("+ .x TestAuth.C(<port>,\"<user>\")                                            +\n");
   printf("+                                                                             +\n");
   printf("+     <port>          = rootd port (default 1094)                             +\n");
   printf("+     <user>          = login user name for the test                          +\n");
   printf("+                      (default from getpwuid)                                +\n");
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

// Details
   TString Details = TString("pt:0 ru:1 us:") + User;

// Testing availabilities
   char *p;

// Test parameter Printout
   printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
   printf("+                                                                             +\n");
   printf("+   Basic test parameters:                                                    +\n");
   printf("+                                                                             +\n");
   printf("+   Local User is          : %s \n",User.Data());
   printf("+   Authentication Details : %s \n",Details.Data());
   printf("+   Current directory is   : %s \n",gSystem->WorkingDirectory());
   printf("+   TFTP string            : %s \n",TFTPPath.Data());
   printf("+                                                                             +\n");
   printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

// Read local <RootAuthrc> now to avoid to be later superseded
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
