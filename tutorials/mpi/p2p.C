#include <Mpi.h>
using namespace ROOT::Mpi;
void p2p()
{
   TEnvironment env; // environment to start communication system

   if (COMM_WORLD.GetSize() != 2) return; // need 2 process

   // data to send/recv
   std::map<std::string, std::string> mymap; // std object
   TMatrixD mymat(2, 2);                     // ROOT object
   Double_t a;                               // default datatype

   // sending messages in process 0
   if (COMM_WORLD.GetRank() == 0) {
      mymap["key"] = "hola";

      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;

      a = 123.0;

      std::cout << "Sending scalar = " << a << std::endl;
      COMM_WORLD.Send(a, 1, 0);
      std::cout << "Sending map = " << mymap["key"] << std::endl;
      COMM_WORLD.Send(mymap, 1, 0);
      std::cout << "Sending mat = ";
      mymat.Print();
      COMM_WORLD.Send(mymat, 1, 0);
   }
   // Receiving messages in process 1
   if (COMM_WORLD.GetRank() == 1) {
      COMM_WORLD.Recv(a, 0, 0);
      std::cout << "Recieved scalar = " << a << std::endl;
      COMM_WORLD.Recv(mymap, 0, 0);
      std::cout << "Received map = " << mymap["key"] << std::endl;
      COMM_WORLD.Recv(mymat, 0, 0);
      std::cout << "Received mat = ";
      mymat.Print();
   }
}
