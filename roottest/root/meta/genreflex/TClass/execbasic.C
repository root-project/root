#include "TClassUtils.h"

//______________________________________________________________________________
int execbasic()
{

   std::cout << "*** Stress TClass ***\n";

   loadLib("libbasic_allClasses_dictrflx");

   // Start the tests
   propertiesNames emptyProp;
   propertiesNames properties1;
   properties1.push_back("checksum");
   printClassInfo("class1",properties1);
   memberNamesProperties class2MemberProp;
   class2MemberProp["m_i"].push_back("property1");
   class2MemberProp["m_i"].push_back("property2");
   printClassInfo("class2",emptyProp,true,class2MemberProp);
   if (gSystem->Getenv("HEADER_PARSING_ON_DEMAND")) {
      gROOT->ProcessLine("#include \"class3.h\"");
   }
   printClassInfo("class3");
   printClassInfo("class3_1");
   printClassInfo("class3_2");
   propertiesNames properties4;
   properties4.push_back("id");
   properties4.push_back("myProp");
   printClassInfo("class4", properties4);
   printClassInfo("class5");
   printClassInfo("class6");

   return 0;
}
