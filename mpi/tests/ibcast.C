#include<Mpi.h>
#include<TMatrixD.h>

using namespace ROOT::Mpi;
void ibcast()
{
    TEnvironment env;
  
    if(gComm->GetSize()==1) return; //needed to run ROOT tutorials in tests

    auto rank=gComm->GetRank();
    auto root=gComm->GetMainProcess();

    //data to send/recv
    TMpiMessage msg;    
    if(gComm->IsMainProcess())
    {
        TMatrixD mymat(2,2);                     //ROOT object
	mymat[0][0] = 0.1;
	mymat[0][1] = 0.2;
	mymat[1][0] = 0.3;
	mymat[1][1] = 0.4;
	msg.WriteObject(mymat);
    }

    auto req=gComm->IBcast(msg,root); //testing TMpiMessage
    req.Complete();
    req.Wait();
    auto mat=(TMatrixD *)msg.ReadObjectAny(TMatrixD::Class());
    
    std::cout<<"Rank = "<<rank<<std::endl;
    mat->Print();
    std::cout.flush();
}


