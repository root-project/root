void pserv(int niter = 100, int bsize = 500000)
{
   // Server program to test parallel sockets.
   //
   // To run this demo do the following:
   //   - Open two windows
   //   - Start ROOT in all two windows
   //   - Execute in the first window: .x pserv.C
   //   - Execute in the second window: .x pclient.C

   // Open a parallel server socket looking for connections on a named
   // service or on a specified port.
   //TPServerSocket *ss = new TServerSocket("rootserv", kTRUE);
   TPServerSocket *ss = new TPServerSocket(9090, kTRUE,
                                           TServerSocket::kDefaultBacklog,
                                           256000);

   // Accept a connection and return a full-duplex communication socket.
   TPSocket *sock = ss->Accept();

   char *buf = new char[bsize];

   // accept data from client
   for (int i = 0; i < niter; i++) {
      memset(buf, 0, bsize);
      int ret = sock->RecvRaw(buf, bsize);
      if (ret < 0) {
         printf("error receiving\n");
         break;
      }
   }

   delete sock;
   delete [] buf;
}
