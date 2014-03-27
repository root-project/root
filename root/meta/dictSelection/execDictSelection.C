
#include "../genreflex/TClass/TClassUtils.h"

//______________________________________________________________________________
int execDictSelection()
{
   
   std::cout << "*** Stress TClass with DictSelection ***\n";
   
   loadLib("libclassesDictSelection_dictrflx.so");

   propertiesNames Nullproperties;
   propertiesNames properties;
   
   // Start the tests
   printClassInfo("classVanilla",Nullproperties,false);   
   printClassInfo("classTemplateVanilla<char>",Nullproperties,false);
   printClassInfo("classTransientMember",Nullproperties,false);
   printClassInfo("classTestAutoselect",Nullproperties,false);
   printClassInfo("classAutoselected",Nullproperties,false);
   printClassInfo("classWithAttributes",Nullproperties,false);
   printClassInfo("classAutoselectedFromTemplateElaborate1",Nullproperties,false);
   printClassInfo("classAutoselectedFromTemplateElaborate2",Nullproperties,false);
// Namespaces not properly treated yet
   printClassInfo("testNs::classRemoveTemplateArgs2<float*>",Nullproperties,false);
   printClassInfo("testNs::classRemoveTemplateArgs2<int*>",Nullproperties,false);
   printClassInfo("testNs::classRemoveTemplateArgs2<classVanilla,int >",Nullproperties,false);
   printClassInfo("A<B<double>,int,float >",Nullproperties,false);
//    printClassInfo("C<char,float>",Nullproperties,false); disabled since the system cannot handle default args
   printClassInfo("C<char>",Nullproperties,false);


   printClassInfo("myVector<C<char>>>",Nullproperties,false);
   printClassInfo("myVector<float>",Nullproperties,false);   

   printClassInfo("classTemplateElaborate<char>",properties,false);
   
   
   return 0;
}
