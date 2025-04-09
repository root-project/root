#include <string>

class MyOverloadOneWay {
public:
   int gime() const {
      return 1;
   }

   std::string gime() {
      return "aap";
   }

};


class MyOverloadTheOtherWay {
public:
   std::string gime() {
      return "aap";
   }

   int gime() const {
      return 1;
   }

};
