#include "xValIncludes.h"

class MyObj{
#include "MyObjBody.h"
};

class MyTest{
#include "MyTestBody.h"
};

int readPtrColls(const char* fname="ptrColls.root")
{
    TFile f(fname);
    MyTest* obj;
    f.GetObject("obj",obj);

    MyTest ref(123);
    int retcode = 0;
    if (ref.ToString() != obj->ToString()) {
    	std::cerr << "The string representations of the objects differ!" << endl
    	          << " - ref: " << ref.ToString() << endl
    	          << " - obj: " << obj->ToString() << endl;
    	retcode=1;
    }

    f.Close();
    return retcode;
}
