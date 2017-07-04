#include <Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

struct particle {
   int x;
   int y;
   int z;
};

void p2p_nonblocking_scalar()
{

   auto size = COMM_WORLD.GetSize();
   if (COMM_WORLD.GetSize() == 1) return; // need at least 2 process

   // data to send/recv
   std::map<std::string, std::string> mymap; // std oebjct
   TMatrixD mymat(2, 2);                     // ROOT object
   particle p;                               // custom object

   if (COMM_WORLD.IsMainProcess()) {
      mymap["key"] = "hola";

      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;

      p.x = 1;
      p.y = 2;
      p.z = 3;

      std::cout << "Sending particle = " << Form("p.x = %d p.y = %d p.x = %d", p.x, p.y, p.z) << std::endl;
      auto req = COMM_WORLD.ISend(p, 1, 0);
      req.Wait();
      std::cout << "Sending map = " << mymap["key"] << std::endl;
      req = COMM_WORLD.ISsend(mymap, 1, 1);
      req.Wait();
      std::cout << "Sending mat = ";
      mymat.Print();
      req = COMM_WORLD.ISend(mymat, 1, 2);
      req.Wait();

      std::cout << "Sending scalar = " << size << std::endl;
      req = COMM_WORLD.ISend(size, 1, 3);
      req.Wait();

   } else {
      // you can Received the messages in other order(is nonblocking)
      auto req = COMM_WORLD.IRecv(mymap, 0, 1);
      req.Wait();
      std::cout << "Received map = " << mymap["key"] << std::endl;

      req = COMM_WORLD.IRecv(mymat, 0, 2);
      req.Wait();
      std::cout << "Received mat = ";
      mymat.Print();
      Int_t scalar;
      req = COMM_WORLD.IRecv(scalar, 0, 3);
      req.Wait();
      std::cout << "Received scalar = " << scalar << std::endl;

      req = COMM_WORLD.IRecv(p, 0, 0);
      req.Wait();
      std::cout << "Received particle = " << Form("p.x = %d p.y = %d p.x = %d", p.x, p.y, p.z) << std::endl;

      TMatrixD req_mat(2, 2);
      req_mat[0][0] = 0.1;
      req_mat[0][1] = 0.2;
      req_mat[1][0] = 0.3;
      req_mat[1][1] = 0.4;

      // assertions
      assert(mymat == req_mat);
      assert(mymap["key"] == "hola");
      assert(p.x == 1);
      assert(p.y == 2);
      assert(p.z == 3);
   }
}

void p2p_nonblocking_array(Int_t count = 2)
{
   auto size = COMM_WORLD.GetSize();
   auto rank = COMM_WORLD.GetRank();

   if (size == 1) return; // need at least 2 process

   // data to send/recv
   std::map<std::string, std::string> mymap[count]; // std oebjct
   TMatrixD mymat[count];                           // ROOT object
   particle p[count];                               // custom object

   TMpiMessage msgs[count];
   TRequest req[4];

   // Testing TMpiMessage
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
      req[0] = COMM_WORLD.ISend(p, count, 1, 4);

      std::cout << "Sending msgs  \n";
      req[1] = COMM_WORLD.ISend(msgs, count, 1, 3);

      std::cout << "Sending maps[\"key\"] = " << mymap[0]["key"] << std::endl;
      req[2] = COMM_WORLD.ISsend(mymap, count, 1, 2);

      std::cout << "Sending mats = ";
      mymat[0].Print();
      req[3] = COMM_WORLD.ISend(mymat, count, 1, 1);

      TRequest::WaitAll(4, req);
   } else {
      // you can Received the messages in other order(is nonblocking)
      req[0] = COMM_WORLD.IRecv(mymap, count, 0, 2);
      req[1] = COMM_WORLD.IRecv(msgs, count, 0, 3);
      req[2] = COMM_WORLD.IRecv(mymat, count, 0, 1);
      req[3] = COMM_WORLD.IRecv(p, count, 0, 4);

      TRequest::WaitAll(4, req);
      std::cout << "Received mat = ";
      mymat[0].Print();

      std::cout << "Received particle = " << Form("p.x = %d p.y = %d p.x = %d", p[0].x, p[0].y, p[0].z) << std::endl;

      std::cout << "Received map = " << mymap[0]["key"] << std::endl;

      TMatrixD req_mat(count, count);
      req_mat[0][0] = 0.1;
      req_mat[0][1] = 0.2;
      req_mat[1][0] = 0.3;
      req_mat[1][1] = 0.4;

      for (auto i = 0; i < count; i++) {
         //          particle &pp= *(particle*)msgs[i].ReadObjectAny(gROOT->GetClass(typeid(particle)));

         // assertions
         assert(mymat[i][0][0] == req_mat[0][0]);
         assert(mymat[i][0][1] == req_mat[0][1]);
         assert(mymat[i][1][0] == req_mat[1][0]);
         assert(mymat[i][1][1] == req_mat[1][1]);
         assert(mymap[i]["key"] == "hola");
         assert(p[i].x == 1);
         assert(p[i].y == 2);
         assert(p[i].z == 3);
      }
   }
}

void p2p_nonblocking()
{
   TEnvironment env; // environment to start communication system

   p2p_nonblocking_scalar();
   p2p_nonblocking_array();
}
