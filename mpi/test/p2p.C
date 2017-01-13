#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void p2p_scalar(TCommunicator &comm)
{
   //data to send/recv
   std::map<std::string, std::string> mymap; //std oebjct
   TMatrixD mymat(2, 2);                    //ROOT object
   Double_t a;                              //default datatype

   if (comm.IsMainProcess()) {
      mymap["key"] = "hola";

      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;

      a = 123.0;

      std::cout << "Sending scalar = " << a << std::endl;
      comm.Send(a, 1, 0);
      std::cout << "Sending map = " << mymap["key"] << std::endl;
      comm.Send(mymap, 1, 0);
      std::cout << "Sending mat = ";
      mymat.Print();
      comm.Send(mymat, 1, 0);
   } else {
      comm.Recv(a, 0, 0);
      std::cout << "Recieved scalar = " << a << std::endl;
      comm.Recv(mymap, 0, 0);
      std::cout << "Received map = " << mymap["key"] << std::endl;
      comm.Recv(mymat, 0, 0);
      std::cout << "Received mat = ";
      mymat.Print();
      TMatrixD req_mat(2, 2); //required mat
      req_mat[0][0] = 0.1;
      req_mat[0][1] = 0.2;
      req_mat[1][0] = 0.3;
      req_mat[1][1] = 0.4;

      //assertions
      assert(a == 123.0);
      assert(mymap["key"] == "hola");
      assert(mymat == req_mat);
   }
}

void p2p_array(TCommunicator &comm)
{
   auto count = 5000;
   TVectorD vecs[count];
   Int_t arr[count];
   if (comm.IsMainProcess()) {
      for (auto i = 0; i < count; i++) {
         vecs[i].ResizeTo(1);
         vecs[i][0] = 1.0;
         arr[i] = i;
      }
      comm.Send(vecs, count, 1, 1);
      comm.Send(arr, count, 1, 1);
   } else {
      comm.Recv(vecs, count, 0, 1);
      comm.Recv(arr, count, 0, 1);
      for (auto i = 0; i < count; i++) {
         vecs[i].Print();
         assert(vecs[i][0] == 1.0);
         assert(arr[i] == i);

      }
   }

}

void p2p()
{
   TEnvironment env;          //environment to start communication system
   TCommunicator comm;   //Communicator to send/recv messages
   p2p_scalar(comm);
   p2p_array(comm);
}

