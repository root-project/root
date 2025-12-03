
#include <TObject.h>
#include <iostream>

class ForeignData
 {
  public:
    ForeignData( int i =0 ) : i_(i)
     { std::cout<<"ForeignData "<<i_<<std::endl ; }
    ForeignData & operator=( int i ) { i_ = i ; return (*this) ; }
    int value() { return i_ ; }
    ~ForeignData()
     { std::cout<<"~ForeignData "<<i_<<std::endl ; }
  private:
    int i_ ;
    ClassDef(ForeignData,1)
 } ;

template <class T>
class CArray : public TObject
 {
  public:

    CArray() : n_objs_(0), objs_(0)
     { std::cout<<"CArray[0]"<<std::endl ; }

    CArray( UInt_t n_objs ) : n_objs_(n_objs)
     {
      std::cout<<"CArray["<<n_objs<<"]"<<std::endl ;
      objs_ = new T[n_objs_] ;
     }

    ~CArray() 
     { 
      delete [] objs_ ;
      std::cout<<"~CArray"<<std::endl ;
     }
     
    T & operator[]( UInt_t i )
     { return objs_[i] ; }

    const T & operator[]( UInt_t i ) const
     { return objs_[i] ; }
    
  private:

// for testing on windows only:
public:
    UInt_t n_objs_ ;
    T * objs_ ; //[n_objs_]

    ClassDefOverride(CArray,1)
 } ;






