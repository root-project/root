#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

struct particle {
   int x;
   int y;
   int z;
};

void p2p_nonblocking_scalar()
{
   TCommunicator comm;        //Communicator to send/recv messages

   if (comm.GetSize() == 1) return; //need at least 2 process


   //data to send/recv
   std::map<std::string, std::string> mymap; //std oebjct
   TMatrixD mymat(2, 2);                    //ROOT object
   particle  p;                             //custom object

   if (comm.IsMainProcess()) {
      mymap["key"] = "hola";

      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;

      p.x = 1;
      p.y = 2;
      p.z = 3;

      std::cout << "Sending particle = " << Form("p.x = %d p.y = %d p.x = %d", p.x, p.y, p.z) << std::endl;
      auto req = comm.ISend(p, 1, 0);
      req.Wait();
      std::cout << "Sending map = " << mymap["key"] << std::endl;
      req = comm.ISsend(mymap, 1, 1);
      req.Wait();
      std::cout << "Sending mat = ";
      mymat.Print();
      req = comm.IRsend(mymat, 1, 2);
      req.Wait();


   } else {
      //you can Received the messages in other order(is nonblocking)
      auto req = comm.IRecv(mymap, 0, 1);
      req.Complete();
      req.Wait();
      std::cout << "Received map = " << mymap["key"] << std::endl;

      req = comm.IRecv(mymat, 0, 2);
      req.Complete();
      req.Wait();
      std::cout << "Received mat = ";
      mymat.Print();

      req = comm.IRecv(p, 0, 0);
      req.Complete();
      req.Wait();
      std::cout << "Received particle = " << Form("p.x = %d p.y = %d p.x = %d", p.x, p.y, p.z) << std::endl;

      TMatrixD req_mat(2, 2);
      req_mat[0][0] = 0.1;
      req_mat[0][1] = 0.2;
      req_mat[1][0] = 0.3;
      req_mat[1][1] = 0.4;


      //assertions
      assert(mymat == req_mat);
      assert(mymap["key"] == "hola");
      assert(p.x == 1);
      assert(p.y == 2);
      assert(p.z == 3);
   }
}

void p2p_nonblocking_array()
{
   TCommunicator comm;        //Communicator to send/recv messages

   if (comm.GetSize() == 1) return; //need at least 2 process
   particle p;
   TGrequest req;
   if (comm.IsMainProcess()) {

      p.x = 1;
      p.y = 2;
      p.z = 3;

      TMpiMessage msg;
      msg.WriteObject(p);
      req = comm.ISend(&msg, 1, 1, 0);
      req.Wait();
   } else {
      TMpiMessage msg;
      req = comm.IRecv(&msg, 1, 0, 0);
      req.Complete();
      req.Wait();

      particle pp = *(particle *)msg.ReadObjectAny(gROOT->GetClass(typeid(particle)));
      assert(pp.x == 1);
      assert(pp.y == 2);
      assert(pp.z == 3);

   }
}


void p2p_nonblocking()
{
   TEnvironment env;          //environment to start communication system

   p2p_nonblocking_scalar();
   p2p_nonblocking_array();
}