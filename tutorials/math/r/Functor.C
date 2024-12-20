/// \file
/// \ingroup tutorial_r
/// \notebook -nodraw
/// Example to create class Functor
///
/// \macro_code
///
/// \author Omar Zapata

#include<TRInterface.h>
#include<TMath.h>

typedef Double_t (*Function)(Double_t);

//Functor class with the function inside
class MyFunctor{
public:
   MyFunctor() {
      status=false;
      f=TMath::BesselY1;
   }

   void setFunction(Function fun) {
    f=fun;
    status=true;
   }

   Bool_t getStatus(){return status;}

   Double_t doEval(Double_t x) {
      return f(x);
   }

private:
   Function f;
   Bool_t status;
};

// this macro exposes the class into R's enviornment
// and lets you pass objects directly.
ROOTR_EXPOSED_CLASS(MyFunctor)

// Macro to create a module
ROOTR_MODULE(MyFunctorModule) {
   ROOT::R::class_<MyFunctor>( "MyFunctor" )
   //creating a default constructor
   .constructor()
   //adding the method doEval to evaluate the internal function
   .method( "doEval", &MyFunctor::doEval )
   .method( "getStatus", &MyFunctor::getStatus)
   ;
}

void Functor()
{
   ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();

   // Creating functor with deafult function TMath::BesselY1
   // and status false from R's environment
   // Loading module into R's enviornment
   r["MyFunctorModule"]<<LOAD_ROOTR_MODULE(MyFunctorModule);

   //creating a class variable from module
   r<<"MyFunctor <- MyFunctorModule$MyFunctor";
   //creating a MyFunctor's object
   r<<"u <- new(MyFunctor)";

   //printing status
   r<<"print(u$getStatus())";

   //printing values from Functor and Function
   r<<"print(sprintf('value in R = %f',u$doEval( 1 )))";
   std::cout<<"value in ROOT = "<<TMath::BesselY1(1)<<std::endl;

   // creating a MyFunctor's object and passing objects to R's
   // enviornment, the status should be true because it is not
   // using the default function
   MyFunctor functor;
   functor.setFunction(TMath::Erf);
   r["functor"]<<functor;

   //printing the status that should be true
   r<<"print(functor$getStatus())";
   r<<"print(sprintf('value in R = %f',functor$doEval( 1 )))";
   std::cout<<"value in ROOT = "<<TMath::Erf(1)<<std::endl;
}
