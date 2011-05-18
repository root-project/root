//--------------------------------------------------
#include "TPServerSocket.h"

//
// This macro should be run together with authclient.C to test
// authentication between two remote ROOT sessions. 
// Run first the authserv.C within a ROOT session on the server
// machine, eg. "srv.machi.ne":
//
//          root[] .x authserv.C(3000)
//
// authserv accepts as argument the port wher it starts listening
// (default 3000).
// You can then run authclient.c in a ROOT session on the client
// machine:
//          root[] .x authclient.C("srv.machi.ne:3000")
//
// and you should get prompted for the credentials, if the case.
// To start a parallel socket of size, for example, 5, enter the
// size as second argument, ie
//
//          root[] .x authclient.C("srv.machi.ne:3000",5)
//

int authserv(int po = 3000)
{

   UChar_t oauth = kSrvAuth;

   TServerSocket *ss = 0;
   TSocket *s = 0;

   cout << "authserv: starting a (parallel) server socket on port "
        << po << " with authentication" << endl;
 
   ss = new TPServerSocket(po);

   // Get the connection
   s = ss->Accept(oauth);

   // Print out;
   if (s) 
      if (s->IsAuthenticated()) 
         cout << "authserv: srv auth socket: OK" << endl;
      else
         cout << "authserv: srv auth socket: failed" << endl;

   // Cleanup
   if (s) delete s;
   if (ss) delete ss;
}
//--------------------------------------------------

