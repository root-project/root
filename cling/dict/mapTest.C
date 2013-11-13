#include <map>
#include "MyTemplateTestClass.h"
#include "RConfig.h"

void mapTest()
{

  
  //std::map<char,int> mymap;
  std::map<char,int>::iterator it;
  std::map<char,int> mymap;

  mymap['a']=50;
  mymap['b']=100;
  mymap['c']=150;
  mymap['d']=200;

  it=mymap.find('b');
  //mymap.erase(it);
  //mymap.erase(mymap.find('d'));

  /*TODO: CINT doesn't interpret the opearor -> properly
  // print content:
  std::cout << "elements in mymap:" << std::endl;
  std::cout << "a => " << mymap.find('a')->second << std::endl;
  std::cout << "c => " << mymap.find('c')->second << std::endl;

  */

  
  MyTemplateClass<int> mtc(-5);
  std::map<int, MyTemplateClass<int> > tmpmap;
#ifdef R__WIN32
  tmpmap.insert( std::pair<const int, MyTemplateClass<int> >(1,mtc) );
#else
  tmpmap.insert( std::pair<int, MyTemplateClass<int> >(1,mtc) );
#endif

  
  /*TODO: CINT doesn't interpret the opearor -> properly
   std::map<int, MyTemplateClass<int> >::iterator mapIt;

  //mapIt = tmpmap.find(1);

  for ( mapIt = tmpmap.begin() ; mapIt != tmpmap.end(); mapIt++ )
  {
    std::cout << mapIt->first << std::endl;
  }
  */
}
