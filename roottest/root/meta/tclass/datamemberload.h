#pragma once

#include <memory>

class Transient {
   int fContent;
};

class Top {
public:
   std::shared_ptr<int> fCounter; ///<! transient but with data members.
   Transient fTransientValue;
   int fValue;
};

class Bottom : public Top {
public:
   Top inner;
   int fOtherValue;
};
