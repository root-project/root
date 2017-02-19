#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;


void port()
{
   TEnvironment env;          //environment to start communication system

   TPort port;
   port.Print();
   assert(port.IsOpen() == kTRUE);
   port.Close();
   port.Print();
   assert(port.IsOpen() == kFALSE);
   port.Open();
   assert(port.IsOpen() == kTRUE);
   port.Print();
   if (COMM_WORLD.GetRank() == 0) {
      port.PublishName("test");
      assert(port.GetPublishName() == "test");
      COMM_WORLD.Send(port.GetPortName(), 1, 0); //sending port to compare with port from LookupName

   } else if (COMM_WORLD.GetRank() == 1) {
      auto ptest = TPort::LookupName("test");
      TString pname;
      COMM_WORLD.Recv(pname, 0, 0);
      assert(pname == ptest.GetPortName());
   }
   //TODO: test Connect/Accept/Disconnect and COMM_SELF
}

Int_t main()
{
   port();
   return 0;
}
