
#include "../genreflex/TClass/TClassUtils.h"
#include "utils.h"

//______________________________________________________________________________
int execDictSelection()
{
   
   std::cout << "*** Stress TClass with DictSelection ***\n";
   
   loadLib("libclassesDictSelection_dictrflx");

   propertiesNames Nullproperties;
   propertiesNames properties;
   
   std::vector<std::string> classNames={
      "classVanilla",
      "classTemplateVanilla<char>",
      "classTransientMember",
      "classTestAutoselect",
      "classAutoselected",
      "classWithAttributes",
      "classAutoselectedFromTemplateElaborate1",
      "classAutoselectedFromTemplateElaborate2",
      "classRemoveTemplateArgs<float,bool>",
      "classRemoveTemplateArgs<testNs::D>",
      "testNs::classRemoveTemplateArgs2<float*>",
      "testNs::classRemoveTemplateArgs2<float*,int,classVanilla>",
      "testNs::classRemoveTemplateArgs2<int*>",
      "testNs::classRemoveTemplateArgs2<int*,int,classVanilla>",
      "testNs::classRemoveTemplateArgs2<classVanilla>",
      "testNs::classRemoveTemplateArgs2<classVanilla,int >",
      "testNs::classRemoveTemplateArgs2<classVanilla,int, classVanilla >",
      "testNs::classRemoveTemplateArgs2<classRemoveTemplateArgs<classAutoselected> >",
      "testNs::classRemoveTemplateArgs2<testNs::D>",
      "A<B<double,double>,int,float >",
      "A<B<double>,int,float >", 
      "C<char>",
      "C<char,int,3>",
      "C<C<char>,C<char,C<C<char,int>,int>,3>>",
      "C<C<C<C<C<char>,C<char,C<C<char,int>,int>,3>>>,C<char,C<C<char,int>,int>,3>>,C<char,C<C<char,int>,int>,3>>",
      "myVector<C<char>>",
      "myVector<C<char,int>>",
      "myVector<C<char,int,3>>",
      "myVector<float>",
      "myVector<myVector<myVector<myVector<float>>>>",  
      "myVector<float,myAllocator<float>>",
      "classTemplateElaborate<char>"
   };

   // Start the tests
   for (auto& className : classNames)
      printClassInfo(className,Nullproperties,false);
   
   // Now some plain name checking
   for (auto& className : classNames)
      printNames(className);

   return 0;
}
