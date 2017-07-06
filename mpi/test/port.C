#include <cassert>
using namespace ROOT::Mpi;

void port()
{
   TEnvironment env; // environment to start communication system

   TPort port;
   port.Print();
   assert(port.IsOpen() == kTRUE);
   port.Close();
   port.Print();
   assert(port.IsOpen() == kFALSE);
   port.Open();
   assert(port.IsOpen() == kTRUE);
   port.Print();
   TPort pp1, pp2;
   assert(pp1 != pp2);

   // TODO: test Connect/Accept/Disconnect and COMM_SELF Publish/UnPublish/LookupName
}

Int_t main()
{
   port();
   return 0;
}
