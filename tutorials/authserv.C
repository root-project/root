//--------------------------------------------------
#include "TPServerSocket.h"

//
// This macro should be run together with authclient.C to test
// authentication between two remote ROOT sessions. 
// Run first the authserv.C within a ROOT session on the server
// machine, eg. "srv.machi.ne"; authserv accepts as argument
// the port wher it starts listening (default 3000).
// You can then run authclient.c in a ROOT session on the client
// machine:
//          root[] .x tutorials.C("srv.machi.ne:3000")
//
// and you should get prompted for the credentials, if the case.
// To start a parallel socket of size, for example, 5, enter the
// size as second argument, ie
//
//          root[] .x tutorials.C("srv.machi.ne:3000",5)
//

int authserv(int po = 3000)
{

   UChar_t oauth = kSrvAuth;

   TServerSocket *ss = 0;
   TSocket *s = 0;

   char buf[256];
   sprintf(buf,"authserv: starting a (parallel) server socket"
               " on port %d with authentication",po);
   cout << buf << endl;
 
   ss = new TPServerSocket(po);

   // Get the connection
   s = ss->Accept(oauth);

   // Print out;
   if (s) 
      if (s->IsAuthenticated()) 
         Printf("authserv: srv auth socket: OK");
      else
         Printf("authserv: srv auth socket: failed");

   // Cleanup
   if (s) delete s;
   if (ss) delete ss;
}
//--------------------------------------------------

