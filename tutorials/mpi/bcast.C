#include<Mpi.h>
#include<TMatrixD.h>

using namespace ROOT::Mpi;
void bcast()
{
  TEnvironment env;
  TCommunicator comm;

  //data to send/recv
  TMatrixD mymat(2,2);                     //ROOT object
  
  auto rank=comm.GetRank();
  auto root=comm.GetMainProcess();
    if(comm.IsMainProcess())
    {
	mymat[0][0] = 0.1;
	mymat[0][1] = 0.2;
	mymat[1][0] = 0.3;
	mymat[1][1] = 0.4;
    }

    comm.Bcast(mymat,root);
    std::cout<<"Rank = "<<rank<<std::endl;
    mymat.Print();
    std::cout.flush();
}


