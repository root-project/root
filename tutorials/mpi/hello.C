using namespace ROOT::Mpi;
void hello()
{
   TEnvironment env; // environment to start communication system
   std::cout << "Hello from process " << COMM_WORLD.GetRank() << " of " << COMM_WORLD.GetSize() << " in host "
             << env.GetProcessorName() << std::endl;
}
