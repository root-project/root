#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;
#define debug  std::cout<<__LINE__<<std::endl;


void commspawn(const Char_t *macropath="./commspawnproc.C")
{
   TEnvironment env;          //environment to start communication system

   if (COMM_WORLD.GetSize() != 4) {
      Error(__FUNCTION__, "needs 4 proccessors");
      COMM_WORLD.Abort(1); //needs 4 ranks
   }
   TNullCommunicator  ncomm; //null comm
   TIntraCommunicator comm;   //default must be null
   TInterCommunicator itcomm;   //default must be null
   TIntraCommunicator gcomm(COMM_WORLD);   //default

   assert(ncomm == comm);
   assert(ncomm == COMM_NULL);
   assert(comm == COMM_NULL);
   assert(gcomm == COMM_WORLD);
   assert(itcomm == COMM_NULL);

   comm = gcomm;
   assert(comm == COMM_WORLD);
   assert(comm.IsInter() == kFALSE); //is an intra comm not an inter comm

   TGroup group = comm.GetGroup();

   const Int_t n = 2;
   const Int_t ranks[n] = {0, 2};

   // pair ranks
   auto pgroup = group.Include(n, ranks);
   assert(pgroup != GROUP_NULL);

   // odd ranks
   auto ogroup = group.Exclude(n, ranks);
   assert(ogroup != GROUP_NULL);

   auto rank = comm.GetRank();


   Int_t root = 0;
   if (rank % 2 == 0) {
      auto pcomm = comm.Create(pgroup); //Internal inter comm that only works in pair ranks

      assert(pcomm != COMM_NULL);
      assert(pcomm.IsInter() == kFALSE); //is an intra comm not an inter comm

      TInfo info;
      const Char_t *argv[] = {"-l", "-q", macropath , NULL}; //running an extarnal MPI macro
      auto icomm = pcomm.Spawn("root", argv, 2, info, root);
      Int_t value = 0;
      icomm.Recv(value, icomm.GetRank(), 0);
      assert(value == 2);
   } else {
      auto ocomm = comm.Create(ogroup);
      assert(ocomm != COMM_NULL);
      assert(ocomm.IsInter() == kFALSE); //is an intra comm not an inter comm

      TInfo info;
      const Char_t *argv[] = {"-l", "-q", macropath , NULL}; //running an extarnal MPI macro
      auto icomm = ocomm.Spawn("root", argv, 2, info, root);
      Int_t value = 0;
      icomm.Recv(value, ocomm.GetRank(), 0);
      assert(value == 2);
   }
}

