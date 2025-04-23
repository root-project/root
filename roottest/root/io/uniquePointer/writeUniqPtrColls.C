#include "xValIncludes.h"

#define USE_UNIQUE

class MyObj{
#include "MyObjBody.h"
};

class MyTest{
#include "MyTestBody.h"
};

int writeUniqPtrColls()
{
    TFile f("uniqPtrColls.root","RECREATE");
    MyTest o(123);
    f.WriteObject(&o,"obj");
    f.Close();
    return 0;
}
