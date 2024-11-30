/// \file
/// \ingroup tutorial_net
/// Server program to test parallel sockets.
///
/// To run this demo do the following:
///   - Open two windows
///   - Start ROOT in all two windows
///   - Execute in the first window: .x pserv.C
///   - Execute in the second window: .x pclient.C
///
/// \macro_code
///
/// \author Fons Rademakers

void pserv()
{
   // Open a parallel server socket looking for connections on a named
   // service or on a specified port.
   //TPServerSocket *ss = new TServerSocket("rootserv", kTRUE);
   TPServerSocket *ss = new TPServerSocket(9090, kTRUE);

   // Accept a connection and return a full-duplex communication socket.
   TPSocket *sock = ss->Accept();
   delete ss;

   int niter, bsize;
   sock->Recv(niter, bsize);

   printf("Receive %d buffers of %d bytes over %d parallel sockets...\n",
          niter, bsize, sock->GetSize());

   char *buf = new char[bsize];

   // start timer
   TStopwatch timer;
   timer.Start();

   // accept data from client
   for (int i = 0; i < niter; i++) {
      memset(buf, 0, bsize);
      int ret = sock->RecvRaw(buf, bsize);
      if (ret < 0) {
         printf("error receiving\n");
         break;
      }
      if (buf[0] != 65) {
         printf("received data corrupted\n");
         break;
      }
   }

   delete sock;
   delete [] buf;

   // stop timer and print results
   timer.Stop();
   Double_t rtime = timer.RealTime();
   Double_t ctime = timer.CpuTime();

   printf("%d bytes received in %f seconds\n", niter*bsize, rtime);
   if (rtime > 0) printf("%5.2f MB/s\n", Double_t(niter*bsize/1024/1024)/rtime);
}
