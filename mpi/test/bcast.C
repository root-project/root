#include<Mpi.h>
#include<TMatrixD.h>
#include <cassert>
#include<particle.h>
using namespace ROOT::Mpi;


void bcast()
{
   TEnvironment env;

   if (gComm->GetSize() == 1) return; //needed at least 2 process

   auto rank = gComm->GetRank();
   auto root = gComm->GetMainProcess();

   //////////////////////////
   //testing TMpiMessage  //
   /////////////////////////
   TMpiMessage msg;
   if (gComm->IsMainProcess()) {
      TMatrixD mymat(2, 2);                    //ROOT object
      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;
      msg.WriteObject(mymat);
   }

   gComm->Bcast(msg, root); //testing TMpiMessage
   auto mat = (TMatrixD *)msg.ReadObjectAny(TMatrixD::Class());

   std::cout << "Rank = " << rank << std::endl;
   mat->Print();
   std::cout.flush();
   TMatrixD req_mat(2, 2);
   req_mat[0][0] = 0.1;
   req_mat[0][1] = 0.2;
   req_mat[1][0] = 0.3;
   req_mat[1][1] = 0.4;



   /////////////////////////
   //testing custom object//
   /////////////////////////
   Particle<Int_t> p;
   if (gComm->IsMainProcess()) {
      p.Set(1, 2);//if root process fill the particle
   }
   gComm->Bcast(p, root); //testing custom object
   p.Print();

   //assertions
   assert(*mat == req_mat);
   assert(p.GetX() == 1);
   assert(p.GetY() == 2);
}


