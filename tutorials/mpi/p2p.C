#include<Mpi.h>
#include<TMatrixD.h>

using namespace ROOT::Mpi;
void p2p()
{
  TEnvironment env;
  TCommunicator comm;
  
  if(comm.GetSize()==1) return; //needed to run ROOT tutorials in tests

  //data to send/recv
  std::map<std::string,std::string> mymap; //std oebjct
  TMatrixD mymat(2,2);                     //ROOT object
  Double_t a;                              //default datatype
  
  auto rank=comm.GetRank();
  
    if(comm.IsMainProcess())
    {
        mymap["key"]="hola";
	
	mymat[0][0] = 0.1;
	mymat[0][1] = 0.2;
	mymat[1][0] = 0.3;
	mymat[1][1] = 0.4;
	
	a=123.0;
        
	std::cout<<"Sending map = "<<mymap["key"]<<std::endl;
        comm.Send(mymap,1,0);
	
        std::cout<<"Sending mat = ";
	mymat.Print();
        comm.Send(mymat,1,0);
        std::cout<<"Sending scalar = "<<a<<std::endl;
	comm.Send(a,1,0);
    }else{
       
        comm.Recv(mymap,0,0);
        std::cout<<"Received map = "<<mymap["key"]<<std::endl;
	
        comm.Recv(mymat,0,0);
        std::cout<<"Received mat = ";
	mymat.Print();
	
        comm.Recv(a,0,0);
        std::cout<<"Recieved scalar = "<<a<<std::endl;
    }    

}


