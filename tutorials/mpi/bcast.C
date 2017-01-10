#include<Mpi.h>
#include<TMatrixD.h>

void bcast()
{
  ROOT::Mpi::TEnvironment env;
  
  if(gComm->GetSize()==1) return; //needed to run ROOT tutorials in tests

  //data to send/recv
  TMatrixD mymat(2,2);                     //ROOT object
  
  auto rank=gComm->GetRank();
  auto root=gComm->GetMainProcess();
    if(gComm->IsMainProcess())
    {
	mymat[0][0] = 0.1;
	mymat[0][1] = 0.2;
	mymat[1][0] = 0.3;
	mymat[1][1] = 0.4;
    }

    gComm->Bcast(mymat,root);
    std::cout<<"Rank = "<<rank<<std::endl;
    mymat.Print();
    std::cout.flush();
}


