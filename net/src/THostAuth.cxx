// @(#)root/net:$Name:  $:$Id: THostAuth.cxx,v 1.1 2003/08/29 10:38:19 rdm Exp $
// Author: G. Ganis   19/03/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THostAuth                                                            //
//                                                                      //
// Contains details about host-specific authentication methods and the  //
// result of their application.                                         //
// Used by TAuthenticate.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include "TSystem.h"
#include "THostAuth.h"
#include "TAuthDetails.h"


ClassImp(THostAuth)

//______________________________________________________________________________
THostAuth::THostAuth(): TObject()
{
   // Default constructor

   fHost        = "";
   fUser        = "";
   fNumMethods  = 0;
   fMethods     = 0;
   fDetails     = 0;
   fEstablished = 0;
}

//______________________________________________________________________________
THostAuth::THostAuth(const char *host, const char *user, Int_t nmeth,
                     Int_t *authmeth, char **details)
{
   // Create hostauth object.

   int i;

   fHost = host;
   // Check and save the host FQDN ...
   TInetAddress addr = gSystem->GetHostByName(fHost);
   if (addr.IsValid()) {
      fHost = addr.GetHostName();
      if (fHost == "UnNamedHost")
         fHost = addr.GetHostAddress();
   }
   fUser = user;
   if (fUser == "")
      fUser = gSystem->Getenv("USER");
   if (fUser == "") {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         fUser = u->fUser;
      delete u;
   }
   fNumMethods = nmeth;
   fMethods    = new Int_t[nmeth];
   if (authmeth != 0) {
      for (i = 0; i < nmeth; i++) { fMethods[i] = authmeth[i]; }
   }
   fDetails = new TString[nmeth];
   if (details) {
      for (i = 0; i < nmeth; i++) {
         if (details[i]) fDetails[i] = details[i];
      }
   }
   fEstablished = new TList;
}

//______________________________________________________________________________
THostAuth::THostAuth(const char *host, const char *user, Int_t authmeth,
                     const char *details)
{
   // Create hostauth object with one method only.

   fHost = host;
    // Check and save the host FQDN ...
   TInetAddress addr = gSystem->GetHostByName(fHost);
   if (addr.IsValid()) {
     fHost = addr.GetHostName();
     if (fHost == "UnNamedHost")
       fHost = addr.GetHostAddress();
   }
   fUser = user;
   if (fUser == "")
      fUser = gSystem->Getenv("USER");
   if (fUser == "") {
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         fUser = u->fUser;
      delete u;
   }
   fNumMethods  = 1;
   fMethods     = new Int_t[1];
   fMethods[0]  = authmeth;
   fDetails     = new TString[1];
   if (details)
      fDetails[0] = details;
   fEstablished = new TList;
}

//______________________________________________________________________________
void  THostAuth::AddMethod(Int_t meth, const char *details)
{
   // Add new method to the list.

   int i;

   // Save existing info
   Int_t   *tMethods = new Int_t[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { tMethods[i] = fMethods[i]; }
   TString *tDetails = new TString[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { tDetails[i] = fDetails[i]; }

   // Resize arrays
   delete [] fMethods;
   delete [] fDetails;
   fMethods = new Int_t[fNumMethods+1];
   for (i = 0; i < fNumMethods; i++) { fMethods[i] = tMethods[i]; }
   fDetails = new TString[fNumMethods+1];
   for (i = 0; i < fNumMethods; i++) { fDetails[i] = tDetails[i]; }

   // delete temporary arrays
   delete [] tMethods;
   delete [] tDetails;

   // This is the new method
   fMethods[fNumMethods] = meth;
   fDetails[fNumMethods] = details;

   // Increment total number
   fNumMethods++;

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void  THostAuth::RemoveMethod(Int_t meth)
{
   // Remove method 'meth' from the list, if there ...

   int i, k;

   // Make sure we are not empty
   if (fNumMethods == 0) return;

   // Check if 'meth' is in the list
   int j = -1;
   for (i = 0; i < fNumMethods; i++) { if (fMethods[i] == meth) j = i; }
   if (j == -1) return;

   // Save existing info
   Int_t   *tMethods = new Int_t[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { tMethods[i] = fMethods[i]; }
   TString *tDetails = new TString[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { tDetails[i] = fDetails[i]; }

   // Resize arrays
   delete [] fMethods;
   delete [] fDetails;
   fMethods = new Int_t[fNumMethods-1];
   fDetails = new TString[fNumMethods-1];
   k = 0;
   for (i = 0; i < fNumMethods; i++) {
      if (tMethods[i] != meth) {
         fMethods[k] = tMethods[i];
         fDetails[k] = tDetails[i];
         k++;
      }
   }
   // delete temporary arrays
   delete [] tMethods;
   delete [] tDetails;

   // Decrement total number
   fNumMethods--;

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
  THostAuth::~THostAuth()
{
   // The dtor.

   delete [] fMethods;
   delete [] fDetails;
   delete    fEstablished;
}

//______________________________________________________________________________
const char *THostAuth::GetDetails(Int_t level)
{
   // Return authentication details for specified level
   // or "" if the specified level does not exist for this host.

   int i;
   for (i = 0; i < fNumMethods; i++) {
      if (fMethods[i] == level) {
         if (gDebug > 3) Info("GetDetails"," %d: returning fDetails[%d]: %s", level,i,fDetails[i].Data());
         return fDetails[i];
      }
   }
   static const char *empty = " ";
   return empty;
}

//______________________________________________________________________________
void THostAuth::SetDetails(Int_t level, const char *details)
{
   // Set authentication details for specified level.

   int i, jm = -1;
   for (i = 0; i < fNumMethods; i++) {
     if (fMethods[i] == level) { fDetails[i] = details; jm = i; break; }
   }
   // If not in the list, add new method ...
   if (jm == -1) AddMethod(level, details);
}

//______________________________________________________________________________
void THostAuth::Print(Option_t *) const
{
   // Print object content.

   // Method names
   const char *AuthMeth[kMAXSEC] = { "UsrPwd","SRP","Krb5","Globus","SSH","UidGid" };

   Info("Print","+------------------------------------------------------------------+");
   Info("Print","+ Host:%s - User:%s - # of available methods:%d",fHost.Data(),fUser.Data(),fNumMethods);
   int i = 0;
   for (i = 0; i < fNumMethods; i++) {
      Info("Print","+ Method: %d (%s)  Details:%s",fMethods[i],AuthMeth[fMethods[i]],fDetails[i].Data());
   }
   Info("Print","+------------------------------------------------------------------+");
}

//______________________________________________________________________________
void THostAuth::Print(const char *proc)
{
   // Print object content.

   // Method names
   const char *AuthMeth[kMAXSEC] = {"UsrPwd","SRP","Krb5","Globus","SSH","UidGid"};

   Info("Print","%s +------------------------------------------------------------------+",proc);
   Info("Print","%s + Host:%s - User:%s - # of available methods:%d",proc,fHost.Data(),fUser.Data(),fNumMethods);
   int i = 0;
   for (i = 0; i < fNumMethods; i++){
      Info("Print","%s + Method: %d (%s)  Details:%s",proc,fMethods[i],AuthMeth[fMethods[i]],fDetails[i].Data());
   }
   Info("Print","%s +------------------------------------------------------------------+",proc);
}

//______________________________________________________________________________
void THostAuth::PrintEstablished()
{
   // Print info about estalished authentication vis-a-vis of this Host.

   Info("PrintEstablished","+------------------------------------------------------------------------------+");
   Info("PrintEstablished","+ Host:%s - Number of Established Authentications: %d",fHost.Data(),fEstablished->GetSize());

   // Check list
   if (fEstablished->GetSize()>0) {
      TIter next(fEstablished);
      TAuthDetails *ai;
      while ((ai = (TAuthDetails*) next()))
         ai->Print("e");
   }
   Info("PrintEstablished","+------------------------------------------------------------------------------+");
}

//______________________________________________________________________________
void  THostAuth::ReOrder(Int_t nmet, Int_t *fmet)
{
   // Set new order for existing methods according to fmet
   int i, j;

   // Book new arrays
   Int_t   *tMethods = new Int_t[fNumMethods];
   TString *tDetails = new TString[fNumMethods];
   Int_t   *flag     = new Int_t[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { flag[i] = 0; }

   // Copy info in the new order
   int k = 0;
   for (j = 0; j < nmet; j++) {
      int jm = -1;
      for (i = 0; i < fNumMethods; i++) {
         if (fmet[j] == fMethods[i] && flag[i] == 0) {
            tMethods[k] = fMethods[i];
            tDetails[k] = fDetails[i];
            k++;
            jm = i;
            flag[i] = 1;
         }
      }
      if (jm == -1) {
         Warning("ReOrder","Enter: method %d not among the ones stored - ignore ",fmet[j]);
      }
   }
   // Copying methods not listed ... if any
   if (k < fNumMethods) {
      for(i = 0; i < fNumMethods; i++){
         if (flag[i] == 0) {
            tMethods[k] = fMethods[i];
            tDetails[k] = fDetails[i];
            k++;
            flag[i] = 1;
         }
      }
   }

   // Resize arrays
   delete [] fMethods;
   delete [] fDetails;
   fMethods = new Int_t[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { fMethods[i] = tMethods[i]; }
   fDetails = new TString[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { fDetails[i] = tDetails[i]; }

   // delete temporary arrays
   delete [] tMethods;
   delete [] tDetails;

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void  THostAuth::SetFirst(Int_t method)
{
   // Set 'method' to be the first used (if in the list ...).

   Int_t *meth = new Int_t[1], nmet = 1;
   meth[0] = method;

   ReOrder(nmet,meth);
   delete [] meth;

   if (gDebug > 3) Print();
}

//______________________________________________________________________________
void THostAuth::SetFirst(Int_t level, const char *details)
{
   // Set as first method 'level' with authentication 'details'.
   // Faster then AddMethod(method,details)+SetFirst(method).

   int i;

   // Check first if the method is there already
   for (i = 0; i < fNumMethods; i++) {
      if (fMethods[i] == level) {
         SetDetails(level, details);
         SetFirst(level);
         if (gDebug > 1) Print();
         return;
      }
   }

   // If not, added in first position ... Save existing info
   Int_t   *tMethods = new Int_t[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { tMethods[i] = fMethods[i]; }
   TString *tDetails = new TString[fNumMethods];
   for (i = 0; i < fNumMethods; i++) { tDetails[i] = fDetails[i]; }

   // Resize arrays
   delete [] fMethods;
   delete [] fDetails;
   fMethods = new Int_t[fNumMethods+1];
   fDetails = new TString[fNumMethods+1];

   // This method first
   fMethods[0] = level;
   fDetails[0] = details;

   // The others ...
   for (i = 0; i < fNumMethods; i++) { fMethods[i+1] = tMethods[i]; }
   for (i = 0; i < fNumMethods; i++) { fDetails[i+1] = tDetails[i]; }

   // delete temporary arrays
   delete [] tMethods;
   delete [] tDetails;

   // Increment total number
   fNumMethods++;

   if (gDebug > 3) Print();
}
