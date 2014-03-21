
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
   printClassInfo("testNs::classRemoveTemplateArgs<float*>",Nullproperties,false);
   printClassInfo("testNs::classRemoveTemplateArgs<int*>",Nullproperties,false);
   printClassInfo("testNs::classRemoveTemplateArgs<classRemoveTemplateArgs<classAutoselected> >",Nullproperties,false);
   printClassInfo("testNs::classRemoveTemplateArgs<classRemoveTemplateArgs<classAutoselected> >",Nullproperties,false);
   printClassInfo("A<B<double,double>,int,float >",Nullproperties,false);
   printClassInfo("C<char,float>",Nullproperties,false);
   printClassInfo("C<char>",Nullproperties,false);
   printClassInfo("myVector<C<char>,myAllocator<C<char>>>",Nullproperties,false);
   printClassInfo("myVector<float,myAllocator<float>>",Nullproperties,false);   
   
   properties.push_back("nonSplittable");
   printClassInfo("classTemplateElaborate<char>",properties,false);
   
   
   return 0;
}
