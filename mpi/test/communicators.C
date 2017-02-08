#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void communicators()
{
   TEnvironment env;          //environment to start communication system

   if (COMM_WORLD.GetSize() < 4) COMM_WORLD.Abort(1); //needs 4 ranks

   TNullCommunicator  ncomm; //null comm
   TIntraCommunicator comm;   //default must be null
   TIntraCommunicator gcomm(COMM_WORLD);   //default

   assert(ncomm == comm);
   assert(ncomm == COMM_NULL);
   assert(comm == COMM_NULL);
   assert(gcomm == COMM_WORLD);

   comm = gcomm;
   assert(comm == COMM_WORLD);
   assert(comm.IsInter() == kFALSE); //is an intra comm not and inter comm

   TGroup group = comm.GetGroup();

   Int_t n = 2;
   const Int_t ranks[2] = {0, 2};

   // pair ranks
   auto pgroup = group.Include(2, ranks);
   assert(pgroup != GROUP_NULL);

   // non-pair ranks
   auto igroup = group.Exclude(2, ranks);
   assert(igroup != GROUP_NULL);


   auto pcomm = comm.Create(pgroup);
   auto icomm = comm.Create(igroup);

   assert(pcomm != icomm);

   assert(pcomm != COMM_NULL);
   assert(icomm != COMM_NULL);


//    assert(pcomm.IsInter()==kFALSE); //is an intra comm not and inter comm

}

