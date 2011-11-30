#include "TMessage.h"
#include "TBenchmark.h"
#include "TSocket.h"
#include "TH2.h"
#include "TTree.h"
#include "TMemFile.h"
#include "TRandom.h"
#include "TError.h"
#include "TFileMerger.h"

#include "TServerSocket.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TMonitor.h"

#include "TFileCacheWrite.h"

void fastMergeServer(bool cache = false) {
   // This script shows how to make a simple iterative server that
   // can receive TMemFile from multiple clients and merge them into 
   // a single file without block.
   //
   // Note: This server assumes that the client will reset the histogram
   // after each upload to simplify the merging.
   // 
   // This server can accept connections while handling currently open connections.
   // Compare this script to hserv.C that blocks on accept.
   // In this script a server socket is created and added to a monitor.
   // A monitor object is used to monitor connection requests on
   // the server socket. After accepting the connection
   // the new socket is added to the monitor and immediately ready
   // for use. Once two connections are accepted the server socket
   // is removed from the monitor and closed. The monitor continues
   // monitoring the sockets.
   //
   // To run this demo do the following:
   //   - Open three windows
   //   - Start ROOT in all three windows
   //   - Execute in the first window: .x fastMergerServer.C
   //   - Execute in the second and third windows: .x treeClient.C
   //Author: Fons Rademakers
   
   // Open a server socket looking for connections on a named service or
   // on a specified port.
   //TServerSocket *ss = new TServerSocket("rootserv", kTRUE);
   TServerSocket *ss = new TServerSocket(9090, kTRUE);
   if (!ss->IsValid()) {
      return;
   }

   TMonitor *mon = new TMonitor;
   
   mon->Add(ss);

   UInt_t clientCount = 0;
   TMemFile *transient = 0;
   
   TFileMerger merger(kFALSE,kFALSE);
   merger.SetPrintLevel(0);

   enum StatusKind {
      kStartConnection = 0,
      kProtocol = 1,
      
      kProtocolVersion = 1
   };
   if (cache) new TFileCacheWrite(merger.GetOutputFile(),32*1024*1024);
   while (1) {
      TMessage *mess;
      TSocket  *s;

      s = mon->Select();

      if (s->IsA() == TServerSocket::Class()) {
         if (clientCount > 100) {
            printf("only accept 100 clients connections\n");
            mon->Remove(ss);
            ss->Close();         
         } else {
            TSocket *client = ((TServerSocket *)s)->Accept();
            client->Send(clientCount, kStartConnection);
            client->Send(kProtocolVersion, kProtocol);
            ++clientCount;
            mon->Add(client);
            printf("Accept %d connections\n",clientCount);
         }
         continue;
      }
      
      s->Recv(mess);

      if (mess==0) {
         Error("fastMergeServer","The client did not send a message\n");
      } else if (mess->What() == kMESS_STRING) {
         char str[64];
         mess->ReadString(str, 64);
         printf("Client %d: %s\n", clientCount, str);
         mon->Remove(s);
         printf("Client %d: bytes recv = %d, bytes sent = %d\n", clientCount, s->GetBytesRecv(),
                s->GetBytesSent());
         s->Close();
         --clientCount;
         if (mon->GetActive() == 0 || clientCount == 0) {
            printf("No more active clients... stopping\n");
            break;
         }
      } else if (mess->What() == kMESS_ANY) {

         Long64_t length;
         TString filename;
         Int_t clientId;
         mess->ReadInt(clientId);
         mess->ReadTString(filename);
         mess->ReadLong64(length); // '*mess >> length;' is broken in CINT for Long64_t.

         Info("fastMergeServer","Receive input from client %d for %s",clientId,filename.Data());
         
         delete transient;
         transient = new TMemFile(filename,mess->Buffer() + mess->Length(),length);
         mess->SetBufferOffset(mess->Length()+length);
         merger.OutputFile(filename);
         merger.AddAdoptFile(transient);

         merger.PartialMerge(TFileMerger::kAllIncremental);
         transient = 0;
      } else if (mess->What() == kMESS_OBJECT) {
         printf("got object of class: %s\n", mess->GetClass()->GetName());
      } else {
         printf("*** Unexpected message ***\n");
      }

      delete mess;
   }
}
