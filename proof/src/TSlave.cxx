// @(#)root/proof:$Name:  $:$Id: TSlave.cxx,v 1.3 2000/11/21 12:27:59 rdm Exp $
// Author: Fons Rademakers   14/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSlave                                                               //
//                                                                      //
// This class describes a PROOF slave server.                           //
// It contains information like the slaves host name, ordinal number,   //
// performance index, socket, etc. Objects of this class can only be    //
// created via TProof member functions.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSlave.h"
#include "TProof.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TROOT.h"

ClassImp(TSlave)

//______________________________________________________________________________
TSlave::TSlave(const char *host, Int_t port, Int_t ord, Int_t perf,
               const char *image, TProof *proof)
{
   // Create a PROOF slave object. Called via the TProof ctor.

   fName    = host;
   fPort    = port;
   fImage   = image;
   fWorkDir = kPROOF_WorkDir;
   fOrdinal = ord;
   fPerfIdx = perf;
   fProof   = proof;
   fSocket  = 0;
   fInput   = 0;

   // Open connection to remote PROOF slave server.
   fSocket = new TSocket(host, port);
   if (fSocket->IsValid()) {
      // Remove socket from global TROOT socket list. Only the TProof object,
      // representing all slave sockets, will be added to this list. This will
      // ensure the correct termination of all proof servers in case the
      // root session terminates.
      gROOT->GetListOfSockets()->Remove(fSocket);

      // Tell remote server to act as master or slave server
      if (proof->IsMaster())
         fSocket->Send("slave");
      else
         fSocket->Send("master");

      // Send user name and passwd to remote host (use trivial
      // inverted byte encoding)
      char user_pass[68], buf[512];
      sprintf(user_pass, "%s %s", fProof->GetUser(), fProof->fPasswd.Data());

      for (int i = 0; i < (int)strlen(user_pass); i++)
         user_pass[i] = ~user_pass[i];

      fSocket->Send(user_pass);

      fSocket->Recv(buf, sizeof(buf));

      if (strcmp(buf, "Okay")) {
         Printf("%s", buf);
         SafeDelete(fSocket);
      } else {
         // get back startup message of proofserv (we are now talking with
         // the real proofserver and not anymore with the proofd front-end)

         Int_t what;
         fSocket->Recv(buf, sizeof(buf), what);
         Printf("%s", buf);

         if (what == kMESS_NOTOK) {
            SafeDelete(fSocket);
            return;
         }

         if (!fProof->IsMaster())
            sprintf(buf, "%s %s %s %d", fProof->GetUser(), user_pass,
                    fProof->GetConfFile(), fProof->GetProtocol());
         else
            sprintf(buf, "%s %d %d", fProof->GetUser(),
                    fProof->GetProtocol(), fOrdinal);
         fSocket->Send(buf);

         fSocket->SetOption(kNoDelay, 1);
         fSocket->SetOption(kSendBuffer, 65536);
         fSocket->SetOption(kRecvBuffer, 65536);
      }
   } else
      SafeDelete(fSocket);
}

//______________________________________________________________________________
TSlave::~TSlave()
{
   // Destroy slave.

   Close();
}

//______________________________________________________________________________
void TSlave::Close(Option_t *)
{
   // Close slave socket.

   SafeDelete(fInput);
   SafeDelete(fSocket);
}

//______________________________________________________________________________
Int_t TSlave::Compare(TObject *obj)
{
   // Used to sort slaves by performance index.

   TSlave *sl = (TSlave *) obj;

   if (fPerfIdx > sl->GetPerfIdx()) return 1;
   if (fPerfIdx < sl->GetPerfIdx()) return -1;
   return 0;
}

//______________________________________________________________________________
void TSlave::Print(Option_t *)
{
   // Printf info about slave.

   Printf("*** Slave %d  (%s)", fOrdinal, fSocket ? "valid" : "invalid");
   Printf("    Host name:            %s", GetName());
   Printf("    Port number:          %d", GetPort());
   Printf("    Image name:           %s", GetImage());
   Printf("    Working directory:    %s", GetWorkingDirectory());
   Printf("    Performance index:    %d", GetPerfIdx());
   Printf("    MB's processed:       %.2f", float(GetBytesRead())/(1024*1024));
   Printf("    MB's sent:            %.2f", fSocket ? float(fSocket->GetBytesRecv())/(1024*1024) : 0.0);
   Printf("    MB's received:        %.2f", fSocket ? float(fSocket->GetBytesSent())/(1024*1024) : 0.0);
   Printf("    Real time used (s):   %.3f", GetRealTime());
   Printf("    CPU time used (s):    %.3f", GetCpuTime());
}

//______________________________________________________________________________
void TSlave::SetInputHandler(TFileHandler *ih)
{
   // Adopt and register input handler for this slave. Handler will be deleted
   // by the slave.

   fInput = ih;
   fInput->Add();
}
