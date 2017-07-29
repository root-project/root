#include <TMpi.h>
using namespace ROOT::Mpi;

void port()
{
   TEnvironment env; // environment to start communication system

   TPort port;
   port.Print();
   ROOT_MPI_ASSERT(port.IsOpen() == kTRUE);
   port.Close();
   port.Print();
   ROOT_MPI_ASSERT(port.IsOpen() == kFALSE);
   port.Open();
   ROOT_MPI_ASSERT(port.IsOpen() == kTRUE);
   port.Print();
   TPort pp1, pp2;
   ROOT_MPI_ASSERT(pp1 != pp2);

   // TODO: test Connect/Accept/Disconnect and COMM_SELF Publish/UnPublish/LookupName
}

Int_t main()
{
   port();
   return 0;
}
