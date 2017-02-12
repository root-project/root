#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;
#define debug  std::cout<<__LINE__<<std::endl;


void communicators()
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

   Int_t n = 2;
   const Int_t ranks[2] = {0, 2};

   // pair ranks
   auto pgroup = group.Include(2, ranks);
   assert(pgroup != GROUP_NULL);

   // odd ranks
   auto ogroup = group.Exclude(2, ranks);
   assert(ogroup != GROUP_NULL);

   auto rank = comm.GetRank();

   if (rank % 2 == 0) {
      auto pcomm = comm.Create(pgroup); //Internal inter comm that only works in pair ranks

      assert(pcomm != COMM_NULL);
      assert(pcomm.IsInter() == kFALSE); //is an intra comm not an inter comm

      //performing a allreduce for pair group
      Int_t results = 0;
      auto lrank = pcomm.GetRank();
      pcomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 1);

      lrank = comm.GetRank();
      pcomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 2);


   } else {
      auto ocomm = comm.Create(ogroup);
      assert(ocomm != COMM_NULL);
      assert(ocomm.IsInter() == kFALSE); //is an intra comm not an inter comm

      //performing a allreduce for odd group
      Int_t results = 0;
      auto lrank = ocomm.GetRank();
      ocomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 1);

      lrank = comm.GetRank();
      ocomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 4);
   }

//testing TIntraCommunicator
   auto key = comm.GetRank();
   auto color = comm.GetRank() % 2;
   gcomm = comm.Split(color, key);

   assert(gcomm != COMM_NULL);
   assert(gcomm.IsInter() == kFALSE); //is an intra comm not and inter comm


//local
   auto icomm = gcomm.CreateIntercomm(0, comm, 1 - color, 0); //creating an intercomm to communicate with odd ranks

   assert(icomm != COMM_NULL);
   assert(icomm.IsInter() == kTRUE); //is an intra comm not and inter comm


// std::cout<<icomm.GetRank()<<std::endl;
// std::cout<<icomm.GetSize()<<std::endl;

   if (rank % 2 == 0) {
      assert(icomm.GetRank() == 0 || icomm.GetRank() == 1);
      assert(icomm.GetSize() == 2);

      assert(icomm.GetRemoteSize() == 2);

      if (icomm.GetRank() == 0) icomm.Send(comm.GetSize(), 1, 12); //sending value to the second gruop

   } else {
      assert(icomm.GetRank() == 0 || icomm.GetRank() == 1);
      assert(icomm.GetSize() == 2);
      assert(icomm.GetRemoteSize() == 2);

      Int_t value;
      if (icomm.GetRank() == 1) {
         icomm.Recv(value, 0, 12); //recv value from the first gruop
         assert(value == comm.GetSize());
      }
   }

}

