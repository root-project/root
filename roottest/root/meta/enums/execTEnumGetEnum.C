#include "tEnumGetEnumClasses.h"

void checkTypeInfo(const std::type_info& ti){
   if (TEnum* en = TEnum::GetEnum(ti)){
      if (const TEnumConstant* enc = en->GetConstant("enc1")){
         std::cout << "Constant found with value " << enc->GetValue() << std::endl;
      }
   } else {
      std::cout << "Enum not found!\n";
   }

}

int execTEnumGetEnum(){

   checkTypeInfo(typeid(enum1));
   checkTypeInfo(typeid(ns1::enum2));
   checkTypeInfo(typeid(ns1::ns2::enum3));
   checkTypeInfo(typeid(class1::enum4));
   checkTypeInfo(typeid(class3<double,1>::enum6));
   checkTypeInfo(typeid(ns4::class4<std::string,1>::enum6));
   // Not existing
   enum enum100{aa};
   checkTypeInfo(typeid(enum100));

   return 0;

}
