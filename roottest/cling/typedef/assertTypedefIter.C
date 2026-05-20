#include <iostream>

int assertTypedefIter() {

   TypedefInfo_t* t = gInterpreter->TypedefInfo_Factory();
   
   while (gInterpreter->TypedefInfo_Next(t)) {
      cout << gInterpreter->TypedefInfo_Name(t) << " " << gInterpreter->TypedefInfo_TrueName(t) << endl;
   }
   return 0;
}
