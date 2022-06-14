/// \file
/// \ingroup tutorial_net
/// Client program to test parallel sockets.
///
/// To run this demo do the following:
///   - Open two windows
///   - Start ROOT in all two windows
///   - Execute in the first window: .x pserv.C
///   - Execute in the second window: .x pclient.C
/// If you want to run the pserv.C on a different host, just change
/// "localhost" in the TPSocket ctor below to the desired hostname.
///
/// \macro_code
///
/// \author Fons Rademakers

void pclient(int niter = 100, int bsize = 500000, int nsocks = 5)
{
   // Open connection to server
   TPSocket *sock = new TPSocket("localhost", 9090, nsocks);
   //TPSocket *sock = new TPSocket("pcroot2", 9090, nsocks);

   char *buf = new char[bsize];
   memset(buf, 65, bsize);

   sock->Send(niter, bsize);

   // send data to server
   for (int i = 0; i < niter; i++) {
      int ret = sock->SendRaw(buf, bsize);
      if (ret < 0) {
         printf("error sending\n");
         break;
      }
   }

   delete sock;
   delete [] buf;
}
