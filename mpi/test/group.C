#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void group()
{
   TEnvironment env;          //environment to start communication system
   TIntraCommunicator comm(COMM_WORLD);   //Communicator to send/recv messages


   if (comm.GetSize() != 4) comm.Abort(1); //requires a least 4 process


   TGroup group = comm.GetGroup();

   Int_t n = 2;
   const Int_t ranks[2] = {0, 2};



   assert(group.GetSize() == comm.GetSize()); //is the global group

// pair ranks
   auto pgroup = group.Include(2, ranks);

// non-pair ranks
   auto igroup = group.Exclude(2, ranks);

   assert(pgroup.Compare(igroup) == ROOT::Mpi::UNEQUAL);


   if (comm.GetRank() % 2 == 0) { //if rank is pair the I can to use the pgroup
      assert(pgroup.GetSize() == 2);
      assert(pgroup.GetRank() == 0 || pgroup.GetRank() == 1);

      assert(igroup.GetSize() == 2);
      assert(igroup.GetRank() != 0 && igroup.GetRank() != 1);

   } else {
      assert(pgroup.GetSize() == 2);
      assert(pgroup.GetRank() != 0 && pgroup.GetRank() != 1);

      assert(igroup.GetSize() == 2);
      assert(igroup.GetRank() == 0 || igroup.GetRank() == 1);
   }

// putting all ranks toguether,
   auto allgroup = TGroup::Union(pgroup, igroup);
   assert(allgroup.GetSize() == comm.GetSize());

   assert(allgroup.Compare(group) == ROOT::Mpi::SIMILAR);


   if (comm.GetRank() % 2 == 0) { //if rank is pair the I can to use the pgroup
      auto pcomm = comm.Create(pgroup); //Intracomm for pair ranks
      auto grank = pcomm.GetRank(); //rank must be  0 and 1
      auto result = 0;
      pcomm.Reduce(grank, result, SUM, 0);
      if (grank == 0) {
         assert(result == 1);
      }

   } else {
      auto icomm = comm.Create(igroup); //Intracomm for pair ranks
      auto grank = icomm.GetRank();//rank must be  0 and 1
      auto result = 0;
      icomm.Reduce(grank, result, SUM, 0);
//           std::cout<<"gsize = "<<icomm.GetSize()<<std::endl;
//           std::cout<<"grank = "<<icomm.GetRank()<<std::endl;
//           std::cout<<"rsult = "<<result<<std::endl;
      if (grank == 0) assert(result == 1);

   }

   TGroup g;
   assert(g == GROUP_NULL);
}

