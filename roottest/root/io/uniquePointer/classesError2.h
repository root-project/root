#ifndef __CLASSESERROR2_H__
#define __CLASSESERROR2_H__

#include <memory>
#include <string>
#include <vector>
#include "TH1F.h"

class A{
   std::vector<std::unique_ptr<TH1F>> d;
};

#endif
