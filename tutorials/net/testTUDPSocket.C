#include "TUDPSocket.h"
#include "TString.h"

//
// As test echo server use udpserver.c in the same directory.
// To compile it do:
//    clang udpserver.c -o udpserver
//

// int createTServerSocket(){
//    printf("testTSocket: Creating TSocket\n");
//    TServerSocket * fServerSocket = new TServerSocket(1501, 0, 0, -1, "UDP");
//
//    TMonitor *mon = new TMonitor;
//
//    mon->Add(fServerSocket);
//
//    TSocket *s0 = 0;
//
//  //  while (1) {
//       char msgRaw[1024] = {0};
//       TSocket  *s;
//
//       s = mon->Select();
//
//       if (s->IsA() == TServerSocket::Class()) {
//          if (!s0) {
//             s0 = ((TServerSocket *)s)->Accept();
//             s0->Send("go 0");
//             mon->Add(s0);
//          }
//
//          if (s0) {
//             mon->Remove(ss);
//             ss->Close();
//          }
//          continue;
//       }
//
//       s->RecvRaw((void *) msgRaw, 1024);
//       printf("Server Message Received %s\n", msgRaw);
//       s->SendRaw(msgRaw, 1024);
//
// // }/* end of server infinite loop */
//
//    s0->Close();
//
//    return 1;
// }

int testTUDPSocket()
{
   printf("testTSocket: Creating TUDPSocket\n");
   TUDPSocket * fSocket = new TUDPSocket("localhost", 1500);

   if (!fSocket || !fSocket->IsValid()) {
      Error("testTSocket","cannot connect to localhost");
      return -1;
   }

   TString msg = "testTSocket: Testing TSocket with UDP";

   printf("%s\n",msg.Data());

   if (fSocket->SendRaw(msg.Data(), msg.Length()) == -1) {
      Error("testTSocket", "error sending command to host %s", fServer.GetHost());
      return -1;
   }

   char msgRaw[1024] = {0};

   fSocket->SetOption(kNoBlock, 1);
   fSocket->Select();

   Int_t recvBytes = fSocket->RecvRaw(msgRaw, 1024);

   if (recvBytes == -1){
      Error("testTSocket", "error receiving data from host %s", fServer.GetHost());
      return -1;
   }

   printf("Received Message: \n%s\n",msgRaw);

   return 1;
}
