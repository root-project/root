{
   // This script shows how to make a simple iterative server that
   // can accept connections while handling currently open connections.
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
   //   - Execute in the first window: .x hserv2.C
   //   - Execute in the second and third windows: .x hclient.C
   //Author: Fons Rademakers

   // Create canvas and pads to display the histograms
   TCanvas *c1 = new TCanvas("c1","The Ntuple canvas",200,10,700,780);
   TPad *pad1 = new TPad("pad1","This is pad1",0.02,0.52,0.98,0.98,21);
   TPad *pad2 = new TPad("pad2","This is pad2",0.02,0.02,0.98,0.48,21);
   pad1->Draw();
   pad2->Draw();

   // Open a server socket looking for connections on a named service or
   // on a specified port.
   //TServerSocket *ss = new TServerSocket("rootserv", kTRUE);
   TServerSocket *ss = new TServerSocket(9090, kTRUE);

   TMonitor *mon = new TMonitor;

   mon->Add(ss);

   TSocket *s0 = 0, *s1 = 0;

   while (1) {
      TMessage *mess;
      TSocket  *s;

      s = mon->Select();

      if (s->IsA() == TServerSocket::Class()) {
         if (!s0) {
            s0 = ((TServerSocket *)s)->Accept();
            s0->Send("go 0");
            mon->Add(s0);
         } else if (!s1) {
            s1 = ((TServerSocket *)s)->Accept();
            s1->Send("go 1");
            mon->Add(s1);
         } else
            printf("only accept two client connections\n");

         if (s0 && s1) {
            mon->Remove(ss);
            ss->Close();
         }
         continue;
      }

      s->Recv(mess);

      if (mess->What() == kMESS_STRING) {
         char str[64];
         mess->ReadString(str, 64);
         printf("Client %d: %s\n", s==s0 ? 0 : 1, str);
         mon->Remove(s);
         if (mon->GetActive() == 0) {
            printf("No more active clients... stopping\n");
            break;
         }
      } else if (mess->What() == kMESS_OBJECT) {
         //printf("got object of class: %s\n", mess->GetClass()->GetName());
         TH1 *h = (TH1 *)mess->ReadObject(mess->GetClass());
         if (h) {
            if (s == s0)
               pad1->cd();
            else
               pad2->cd();
            h->Print();
            h->DrawCopy();  //draw a copy of the histogram, not the histo itself
            c1->Modified();
            c1->Update();
            delete h;       // delete histogram
         }
      } else {
         printf("*** Unexpected message ***\n");
      }

      delete mess;
   }

   printf("Client 0: bytes recv = %d, bytes sent = %d\n", s0->GetBytesRecv(),
          s0->GetBytesSent());
   printf("Client 1: bytes recv = %d, bytes sent = %d\n", s1->GetBytesRecv(),
          s1->GetBytesSent());

   // Close the socket.
   s0->Close();
   s1->Close();
}
