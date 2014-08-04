void hserv() {
   // Server program which waits for two clients to connect. It then monitors
   // the sockets and displays the objects it receives. To see how to
   // make a non-blocking server see the script hserv2.C.
   //
   // To run this demo do the following:
   //   - Open three windows
   //   - Start ROOT in all three windows
   //   - Execute in the first window: .x hserv.C
   //   - Execute in the second and third windows: .x hclient.C
   //Author: Fons Rademakers

   // Open a server socket looking for connections on a named service or
   // on a specified port.
   //TServerSocket *ss = new TServerSocket("rootserv", kTRUE);
   TServerSocket *ss = new TServerSocket(9090, kTRUE);

   // Accept a connection and return a full-duplex communication socket.
   TSocket *s0 = ss->Accept();
   TSocket *s1 = ss->Accept();

   // tell the clients to start
   s0->Send("go 0");
   s1->Send("go 1");

   // Close the server socket (unless we will use it later to wait for
   // another connection).
   ss->Close();

   // Check some options of socket 0.
   int val;
   s0->GetOption(kSendBuffer, val);
   printf("sendbuffer size: %d\n", val);
   s0->GetOption(kRecvBuffer, val);
   printf("recvbuffer size: %d\n", val);

   // Get the remote addresses (informational only).
   TInetAddress adr = s0->GetInetAddress();
   adr.Print();
   adr = s1->GetInetAddress();
   adr.Print();

   // Create canvas and pads to display the histograms
   TCanvas *c1 = new TCanvas("c1","The Ntuple canvas",200,10,700,780);
   TPad *pad1 = new TPad("pad1","This is pad1",0.02,0.52,0.98,0.98,21);
   TPad *pad2 = new TPad("pad2","This is pad2",0.02,0.02,0.98,0.48,21);
   pad1->Draw();
   pad2->Draw();

   TMonitor *mon = new TMonitor;

   mon->Add(s0);
   mon->Add(s1);

   while (1) {
      TMessage *mess;
      TSocket  *s;

      s = mon->Select();

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
