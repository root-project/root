#include <Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void p2p_scalar(Int_t size = 10)
{
   // data to send/recv
   std::map<std::string, std::string> mymap; // std oebjct
   TMatrixD mymat(size, size);               // ROOT object
   Double_t a;                               // default datatype

   if (COMM_WORLD.IsMainProcess()) {
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
      //       mymat.Print();
      COMM_WORLD.Send(mymat, 1, 0);
   } else if (COMM_WORLD.GetRank() == 1) {
      COMM_WORLD.Recv(a, 0, 0);
      std::cout << "Recieved scalar = " << a << std::endl;
      COMM_WORLD.Recv(mymap, 0, 0);
      std::cout << "Received map = " << mymap["key"] << std::endl;
      COMM_WORLD.Recv(mymat, 0, 0);
      std::cout << "Received mat = ";
      //       mymat.Print();
      TMatrixD req_mat(size, size); // required mat
      req_mat[0][0] = 0.1;
      req_mat[0][1] = 0.2;
      req_mat[1][0] = 0.3;
      req_mat[1][1] = 0.4;

      // assertions
      assert(a == 123.0);
      assert(mymap["key"] == "hola");
      assert(mymat == req_mat);
   }
}

void p2p_array(Int_t count = 500)
{
   TVectorD vecs[count];
   Int_t arr[count];
   if (COMM_WORLD.IsMainProcess()) {
      for (auto i = 0; i < count; i++) {
         vecs[i].ResizeTo(count);
         vecs[i][0] = 1.0;
         arr[i] = i;
      }
      COMM_WORLD.Send(vecs, count, 1, 1);
      COMM_WORLD.Send(arr, count, 1, 1);
   } else if (COMM_WORLD.GetRank() == 1) {
      COMM_WORLD.Recv(vecs, count, 0, 1);
      COMM_WORLD.Recv(arr, count, 0, 1);
      for (auto i = 0; i < count; i++) {
         //          vecs[i].Print();
         assert(vecs[i][0] == 1.0);
         assert(arr[i] == i);
      }
   }
}

void p2p(Bool_t stressTest = kTRUE)
{
   TEnvironment env; // environment to start communication system
   if (!stressTest) {
      p2p_scalar();
      p2p_array();
   } else {
      for (auto i = 0; i < COMM_WORLD.GetSize() * 2; i++) {
         p2p_scalar((i + 1) * 100);
         p2p_array((i + 1) * 100);
      }
   }
}
