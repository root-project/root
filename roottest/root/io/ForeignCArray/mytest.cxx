
#include "MyClasses.h"
#include <TFile.h>
#include <TTree.h>
#include <iostream>

#include "TApplication.h"

int main(int argc, char** argv)
 {
   TApplication theApp("App", &argc, argv);
   {
    CArray<ForeignData> array(3) ;
    array[0] = 1 ;
    array[1] = 2 ;
    array[2] = 3 ;
    TFile f("mytest.root","recreate") ;
    array.Write("array") ;
    f.Close() ;
   }

   {
    TFile f("mytest.root","read") ;
    CArray<ForeignData> * parray ;
    parray = (CArray<ForeignData> *) f.Get("array") ;
    for ( int i=0 ; i<3 ; ++i )
      std::cout<<(*parray)[i].value()<<std::endl ;
   }

  return 0 ;
 }




