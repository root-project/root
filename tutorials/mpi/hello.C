/// \file
/// \ingroup tutorial_mpi
///
///  Simple example to show the process id and the name of the host.
///
/// ~~~{.cpp}
///  rootmpi -np 2 hello.C
/// ~~~
///
///
/// \macro_output
/// \macro_code
///
/// \author Omar Zapata
using namespace ROOT::Mpi;
void hello()
{
   TEnvironment env; // environment to start communication system
   std::cout << "Hello from process " << COMM_WORLD.GetRank() << " of " << COMM_WORLD.GetSize() << " in host "
             << env.GetProcessorName() << std::endl;
}
