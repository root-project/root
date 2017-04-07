//script to test pass function into a module
#include<TRInterface.h>
#include<TMath.h>

std::string hello( std::string who, std::string msg){
    std::string result( "hello " ) ;
    result += who ;
    result += msg;
    return result ;
} 

ROOTR_MODULE(rootr){
    ROOT::R::function( "hello", &hello );
}

void FunModule()
{
  ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
   r.SetVerbose(kFALSE);
   r["rootr"]<<LOAD_ROOTR_MODULE(rootr);
   r<<"print(rootr$hello('world ','ROOTR'))";
}
