
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
//    printClassInfo("testNs::classRemoveTemplateArgs<float*>",Nullproperties,false);
//    printClassInfo("testNs::classRemoveTemplateArgs<int*>",Nullproperties,false);
//    printClassInfo("testNs::classRemoveTemplateArgs<classRemoveTemplateArgs<classAutoselected> >",Nullproperties,false);
//    printClassInfo("testNs::classRemoveTemplateArgs<classRemoveTemplateArgs<classAutoselected> >",Nullproperties,false);
   printClassInfo("A<B<double,double>,int,float >",Nullproperties,false);
   printClassInfo("C<char,float>",Nullproperties,false);
   printClassInfo("C<char>",Nullproperties,false);

// Canot treat template arguments which are template specialisations whith arguments which are 
// other template parameters, like template <class T, class Alloc= myAllocator<T> > class myVector;
//    printClassInfo("myVector<C<char>>>",Nullproperties,false);
//    printClassInfo("myVector<float,>",Nullproperties,false);   
   
   properties.push_back("nonSplittable");
   printClassInfo("classTemplateElaborate<char>",properties,false);
   
   
   return 0;
}
