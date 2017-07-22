/// \file
/// \ingroup tutorial_mpi
///
/// Collect messages from a group of processes.
/// Each process (root process included) sends the contents of its send buffer to the root process.
/// The root process receives the messages and stores them in rank order.
/// The outcome is as if each of the n processes in the group (including the root process) had executed a call to
/// ROOT::Mpi:TCommunicator::Gather
/// to execute this example with 4 processors, do:
///
/// ~~~{.cpp}
///  rootmpi -np 4 gather.C
/// ~~~
///
///
/// \macro_output
/// \macro_code
///
/// \author Omar Zapata

using namespace ROOT::Mpi;

void gather()
{
   TEnvironment env;
   if (COMM_WORLD.GetSize() == 1)
      return; // needed at least 2 process
   auto rank = COMM_WORLD.GetRank();
   auto size = COMM_WORLD.GetSize();

   auto count = 2;
   auto root = COMM_WORLD.GetMainProcess();

   // creating a vector to send and
   // the array of vectors to receiv.
   TVectorD send_vec[count];
   TVectorD *recv_vec;
   for (auto i = 0; i < count; i++) {
      send_vec[i].ResizeTo(1);
      send_vec[i][0] = rank;
   }

   if (rank == root) {
      recv_vec = new TVectorD[size * count];
   }

   COMM_WORLD.Gather(send_vec, count, recv_vec, size * count, root); // testing custom object

   if (rank == root) {
      // just printing all infortaion
      for (auto i = 0; i < size * count; i++) {
         recv_vec[i].Print();
      }

      for (auto i = 0; i < COMM_WORLD.GetSize(); i++) {

         for (auto j = 0; j < count; j++) {
            std::cout << "vec[" << i * count + j << "] = " << recv_vec[i * count + j][0] << " -- " << i << std::endl;
         }
      }
      delete[] recv_vec;
   }
}
