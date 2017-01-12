#include<Mpi.h>
#include<TMatrixD.h>

using namespace ROOT::Mpi;

//custom class
template<class T> class Particle {
   T x;
   T y;
public:
   Particle(T _x = 0, T _y = 0): x(_x), y(_y) {}
   void Set(T _x, T _y)
   {
      x = _x;
      y = _y;
   }
   void Print()
   {
      std::cout << "x = " << x << " y = " << y << std::endl;
      std::cout.flush();
   }
};

void ibcast()
{
   TEnvironment env;

   if (gComm->GetSize() == 1) return; //needed to run ROOT tutorials in tests

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

   auto req = gComm->IBcast(msg, root); //testing TMpiMessage
   req.Complete();
   req.Wait();
   auto mat = (TMatrixD *)msg.ReadObjectAny(TMatrixD::Class());

   std::cout << "Rank = " << rank << std::endl;
   mat->Print();
   std::cout.flush();

   /////////////////////////
   //testing custom object//
   /////////////////////////
   Particle<Int_t> p;
   if (gComm->IsMainProcess()) {
      p.Set(1, 2);
   }
   req = gComm->IBcast(p, root); //testing custom object
   req.Complete();
   req.Wait();
   p.Print();
}


