#include "TBufferJSON.h"
#include <map>
#include <string>

void testJson(std::map<std::string,int> &data, int compact, const char *name)
{
   auto json = TBufferJSON::ToJSON(&data, compact);

   std::cout << name << json << std::endl;

   auto mread = TBufferJSON::FromJSON<std::map<std::string,int>>(json.Data());
   if (!mread) {
      std::cout << "Fail to read map from JSON" << std::endl;
   } else if (data.size() != mread->size()) {
      std::cout << "Mismatch in maps size " << data.size() << "  " << mread->size() << std::endl;
   } else {
      bool isok = true;
      for (auto &item : data) {
         if (mread->at(item.first) != item.second) {
            std::cout << "Failure with field " << item.first << std::endl;
            isok = false;
         }
      }
      if (isok)
         std::cout << "Data matches" << std::endl;
   }

}

void runMap()
{
   // for dictionary
   gSystem->Load("libJsonTestClasses");

   std::map<std::string,int> data;

   for (int n=0;n<10;++n) {
      std::string key = "field";
      key.append(std::to_string(n));
      data[key] = n*7;
   }

   testJson(data, TBufferJSON::kNoCompress, "DEFAULT: ");

   testJson(data, TBufferJSON::kMapAsObject, "OBJECT: ");

   testJson(data, TBufferJSON::kSkipTypeInfo + TBufferJSON::kMapAsObject + TBufferJSON::kNoSpaces, "MINIMAL: ");

}
