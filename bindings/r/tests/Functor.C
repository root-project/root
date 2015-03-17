#include<TRInterface.h>
#include<TMath.h>
 
typedef Double_t (*Function)(Double_t);
 
class MyFunctor{
public:
  MyFunctor(){
    f=TMath::BesselY1;//here is the function that I want.
  }
  Double_t doEval(Double_t x) {
    return f(x);
  }
private:
   Function f;
};
 
ROOTR_MODULE(MyFunctorModule) {
    ROOT::R::class_<MyFunctor>( "MyFunctor" )
    .constructor()
    .method( "doEval", &MyFunctor::doEval )
//  .method( "otherMethod", &MyFunctor::otherMethod )//you can added more methods adding .method(name,pointer)
    ;
}
 
void Functor()
{
   ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
   r["MyFunctorModule"]<<LOAD_ROOTR_MODULE(MyFunctorModule);
 
   r<<"MyFunctor <- MyFunctorModule$MyFunctor";
   r<<"u <- new(MyFunctor)";
   r<<"print(u$doEval( 1 ))";
   std::cout<<"value in ROOT = "<<TMath::BesselY1(1)<<std::endl;
}