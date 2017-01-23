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

   auto size = comm.GetSize();
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

      std::cout << "Sending scalar = " << size << std::endl;
      req = comm.ISend(size, 1, 3);
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
      Int_t scalar;
      req = comm.IRecv(scalar, 0, 3);
//       req.Complete();//disabled for raw types
      req.Wait();
      std::cout << "Received scalar = " << scalar << std::endl;

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

void p2p_nonblocking_array(Int_t count = 10)
{
   auto size = gComm->GetSize();
   auto rank = gComm->GetRank();

   if (size == 1) return; //need at least 2 process


   //data to send/recv
   std::map<std::string, std::string> mymap[count]; //std oebjct
   TMatrixD mymat[count];                    //ROOT object
   particle  p[count];                             //custom object

   TMpiMessage msgs[count];

   //Testing TMpiMessage
   if (rank == 0) {
      p[0].x = 1;
      p[0].y = 2;
      p[0].z = 3;

      TMpiMessage msg;
      msg.WriteObject(p[0]);
      auto req = gComm->ISend(&msg, 1, 1, 0);
      req.Wait();
   } else {
      TMpiMessage msg;
      auto req = gComm->IRecv(&msg, 1, 0, 0);
      req.Complete();
      req.Wait();

      particle pp = *(particle *)msg.ReadObjectAny(gROOT->GetClass(typeid(particle)));
      assert(pp.x == 1);
      assert(pp.y == 2);
      assert(pp.z == 3);

   }

   //Testing TMpiMessage
   if (rank == 0) {
      for (auto i = 0; i < count; i++) {
         mymap[i]["key"] = "hola";
         mymat[i].ResizeTo(count, count);
         mymat[i][0][0] = 0.1;
         mymat[i][0][1] = 0.2;
         mymat[i][1][0] = 0.3;
         mymat[i][1][1] = 0.4;
         p[i].x = 1;
         p[i].y = 2;
         p[i].z = 3;
         msgs[i].WriteObject(p);
      }
      std::cout << "Sending particle = " << Form("p.x = %d p.y = %d p.x = %d", p[0].x, p[0].y, p[0].z) << std::endl;
      auto  req = gComm->ISend(p, count, 1, 4);
      req.Wait();

//       req = gComm->ISend(msgs, count, 1, 3);
//       req.Wait();

      std::cout << "Sending maps[\"key\"] = " << mymap[0]["key"] << std::endl;
      req = gComm->ISsend(mymap, count, 1, 2);
      req.Wait();

      std::cout << "Sending mats = ";
      mymat[0].Print();
      req = gComm->IRsend(mymat, count, 1, 1);
      req.Wait();

   } else {
      //you can Received the messages in other order(is nonblocking)
     auto req = gComm->IRecv(mymap, count, 0, 3);
     req.Complete();
     req.Wait();
     std::cout << "Received map = " << mymap[0]["key"] << std::endl;
     
     req = gComm->IRecv(p, count, 0, 1);
     req.Complete();
     req.Wait();
     std::cout << "Received particle = " << Form("p.x = %d p.y = %d p.x = %d", p[0].x, p[0].y, p[0].z) << std::endl;
     
     req = gComm->IRecv(mymat, count, 0, 4);
     req.Complete();
     req.Wait();
     std::cout << "Received mat = ";
     mymat[0].Print();
     

//      req = gComm->IRecv(msgs, count, 0, 2);
//      req.Complete();
//      req.Wait();

      TMatrixD req_mat(count, count);
      req_mat[0][0] = 0.1;
      req_mat[0][1] = 0.2;
      req_mat[1][0] = 0.3;
      req_mat[1][1] = 0.4;

      for (auto i = 0; i < count; i++) {
         //assertions
//          assert(mymat[i][0][0] == req_mat[0][0]);
//          assert(mymat[i][0][1] == req_mat[0][1]);
//          assert(mymat[i][1][0] == req_mat[1][0]);
//          assert(mymat[i][1][1] == req_mat[1][1]);
//          assert(mymap[i]["key"] == "hola");
         assert(p[i].x == 1);
         assert(p[i].y == 2);
         assert(p[i].z == 3);
      }
   }
}


void p2p_nonblocking()
{
   TEnvironment env;          //environment to start communication system

//    p2p_nonblocking_scalar();
   p2p_nonblocking_array();
}