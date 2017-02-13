#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

//this macro is launched from commspawn and the both macros can have communication through TInterCommunicator object

#define NUM_SPAWNS 2

void commspawnproc()
{
   TEnvironment env;          //environment to start communication system

   if (COMM_WORLD.GetSize() != NUM_SPAWNS) {
      Error(__FUNCTION__, "needs 2 proccessors");
      COMM_WORLD.Abort(1); //needs 2 ranks
   }


   int np = NUM_SPAWNS;
   int errcodes[NUM_SPAWNS];

   auto inter = TCommunicator::GetParent();
   if (inter != COMM_NULL) {
//       std::cout<<"I'm the spawned sending rank="<<inter.GetRank()<<".\n";
      inter.Send(inter.GetSize(), inter.GetRank(), 0);
   } else {
      std::cout << "I'm the parent.\n";
      COMM_WORLD.Abort(ERR_COMM);
   }
   std::cout.flush();
}

