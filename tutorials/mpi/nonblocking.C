#include<Mpi.h>
using namespace ROOT::Mpi;

struct particle
{
  int x;
  int y;
  int z;
};

void nonblocking()
{
  TEnvironment env;          //environment to start communication system
  TCommunicator comm;        //Communicator to send/recv messages
  
  if(comm.GetSize()==1) return; //need at least 2 process

  
  //data to send/recv
  std::map<std::string,std::string> mymap; //std oebjct
  TMatrixD mymat(2,2);                     //ROOT object
  particle  p;                             //custom object
  
    if(comm.IsMainProcess()) 
    {
        mymap["key"]="hola";

	mymat[0][0] = 0.1;
	mymat[0][1] = 0.2;
	mymat[1][0] = 0.3;
	mymat[1][1] = 0.4;
	
	p.x=1;
	p.y=2;
	p.z=3;
        
        std::cout<<"Sending particle = "<<Form("p.x = %d p.y = %d p.x = %d",p.x,p.y,p.z)<<std::endl;
	auto req=comm.ISend(p,1,0);
        req.Wait();
	std::cout<<"Sending map = "<<mymap["key"]<<std::endl;
        req = comm.ISsend(mymap,1,1);
        req.Wait();
	std::cout<<"Sending mat = ";
	mymat.Print();
        req = comm.IRsend(mymat,1,2);
        req.Wait();

    }else{
      //you can Received the messages in other order(is nonblocking)
        auto req=comm.IRecv(mymap,0,1);
	req.Complete();
	req.Wait();
        std::cout<<"Received map = "<<mymap["key"]<<std::endl;

	req=comm.IRecv(mymat,0,2);
	req.Complete();
	req.Wait();
        std::cout<<"Received mat = ";
	mymat.Print();
	
        req=comm.IRecv(p,0,0);
	req.Complete();
	req.Wait();
        std::cout<<"Received particle = "<<Form("p.x = %d p.y = %d p.x = %d",p.x,p.y,p.z)<<std::endl;
    }  
}