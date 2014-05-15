void runFullheaderParsingOnDemand(){
   gDebug=1;


   std::vector<std::string> classNames={"myClass0<E>",
                                        "myClass1<E>",
                                        "myClass1<int>",
                                        "myClass2<E>",
                                        "myClass2<float>",
                                        "myClass3<E>",
                                        "myClass4<E>",
                                        "myClass5<E>",
                                        "myClass6<E>",
                                        "myClass7<E>",
                                        "myClass8<E>"};
//                                         "myClass8<std::vector<E>>",
//                                         "myClass8<std::vector<std::queue<E>>>",
//                                         "myClass9<E>"};
   for (auto& name : classNames){
      std::cout << "Class name " << name << std::endl;
      TClass::GetClass(name.c_str())->GetClassInfo();
   }
}
