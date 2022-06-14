/// \file
/// \ingroup tutorial_net
/// This macro should be run together with authserv.C to test
/// authentication between two remote ROOT sessions.
/// Run first the authserv.C within a ROOT session on the server
/// machine, eg. "srv.machi.ne":
///
///          root[] .x authserv.C(3000)
///
/// authserv accepts as argument the port where it starts listening
/// (default 3000).
/// You can then run authclient.c in a ROOT session on the client
/// machine:
///          root[] .x authclient.C("srv.machi.ne:3000")
///
/// and you should get prompted for the credentials, if the case.
/// To start a parallel socket of size, for example, 5, enter the
/// size as second argument, ie
///
///          root[] .x authclient.C("srv.machi.ne:3000",5)
///
/// \macro_code
///
/// \author

#include "TPSocket.h"

int authclient(const char *host = "up://localhost:3000", int sz = 0)
{
   Int_t par = (sz > 1) ? 1 : 0;

   // Parse protocol, if any
   TString proto(TUrl(host).GetProtocol());
   TString protosave = proto;

   // Get rid of authentication suffix
   TString asfx = proto;
   if (proto.EndsWith("up") || proto.EndsWith("ug")) {
      asfx.Remove(0,proto.Length()-2);
      proto.Resize(proto.Length()-2);
   } else if (proto.EndsWith("s") || proto.EndsWith("k") ||
              proto.EndsWith("g") || proto.EndsWith("h")) {
      asfx.Remove(0,proto.Length()-1);
      proto.Resize(proto.Length()-1);
   }

   // Force parallel (even of size 1)
   TString newurl = "p" + asfx;
   newurl += "://";
   if (strlen(TUrl(host).GetUser())) {
      newurl += TUrl(host).GetUser();
      newurl += "@";
   }
   newurl += TUrl(host).GetHost();
   newurl += ":";
   newurl += TUrl(host).GetPort();

   cout << "authclient: starting a (parallel) authenticated socket at "
        << newurl.Data() << " (size: " << sz << ")" << endl;

   TSocket *s = TSocket::CreateAuthSocket(newurl.Data(),sz);

   // Print out;
   if (s)
      if (s->IsAuthenticated())
         cout << "authclient: auth socket: OK" << endl;
      else
         cout << "authclient: auth socket: failed" << endl;

   // Cleanup
   if (s) {
      // Remove this authentication from the token list to avoid
      // later warnings
      s->GetSecContext()->DeActivate("R");
      delete s;
   }
}
