
#include<TRInterface.h>

using namespace Rcpp ;

class Foo{
    public:
        enum Bla{ FOO, BAR } ;
        
        Foo( double x_, double y_) : x(x_), y(y_){}
        
        Foo* clone(){
            return new Foo( x, y) ;    
        }
        
        double x, y ;
        
        void bla(const Foo& other){
            Rprintf( "efez\n" ) ;   
        }
        
} ;

Foo make_foo(){ return Foo(3, 4) ; }

RCPP_EXPOSED_CLASS(Foo)

RCPP_MODULE(Mod){
    
    class_<Foo>("Foo" )
        .constructor<double,double>() 
        .method( "clone", &Foo::clone )
        
        .field( "x", &Foo::x )
        .field( "y", &Foo::y )
        
        .method( "bla", &Foo::bla )
    ;
    Rcpp::function( "make_foo", &make_foo ) ;
    
}

void ExpClass()
{
  ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
   r["Mod"]<<LOAD_ROOTR_MODULE(Mod);
   
   r<<"Foo <- Mod$make_foo()";
   r<<"Foo$bla(Foo)";
   r<<"f <- Foo$clone()";
   r<<"f$bla(Foo)";
}