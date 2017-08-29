/// \file
/// \ingroup tutorial_mpi
///
/// Sends data from one task to all tasks in a group.
/// This is the inverse operation to ROOT::Mpi::TCommunicator::Gather.
/// An alternative description is that the root sends a message with ROOT::Mpi::TCommunicator::Send.
/// This message is split into n equal segments, the ith segment is sent to the ith process in the group, and each
/// process receives this message as above.
/// The send buffer is ignored for all nonroot processes.
/// to execute this example with 4 processors, do:
///
/// ~~~{.cpp}
///  rootmpi -np 4 scatter.C
/// ~~~
///
///
/// \macro_output
/// \macro_code
///
/// \author Omar Zapata
#include <cassert>
using namespace ROOT::Mpi;

void scatter()
{
   TEnvironment env;
   env.SyncOutput();
   if (COMM_WORLD.GetSize() == 1)
      return; // needed at least 2 process
   auto rank = COMM_WORLD.GetRank();
   auto size = COMM_WORLD.GetSize();

   auto count = 2;
   auto root = COMM_WORLD.GetMainProcess();

   // creating a vector to send and
   // the array of vectors to receiv.
   TVectorD *send_vec;
   if (root == rank) {
      send_vec = new TVectorD[size * count];
      for (auto i = 0; i < COMM_WORLD.GetSize() * count; i++) {
         send_vec[i].ResizeTo(1);
         send_vec[i][0] = i;
      }
   }
   TVectorD recv_vec[count];

   COMM_WORLD.Scatter(send_vec, size * count, recv_vec, count, root); // testing custom object

   for (auto i = 0; i < count; i++) {
      recv_vec[i].Print();
      std::cout << recv_vec[i][0] << " -- " << (rank * count + i) << std::endl;
      // assertions
      assert(recv_vec[i][0] == (rank * count + i));
   }

   if (rank == root)
      delete[] send_vec;
}
