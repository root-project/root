#include "xValIncludes.h"

class MyObj{
#include "MyObjBody.h"
};

class MyTest{
#include "MyTestBody.h"
};

int writePtrColls()
{
    TFile f("ptrColls.root","RECREATE");
    MyTest o(123);
    f.WriteObject(&o,"obj");
    f.Close();
    return 0;
}
