#include "testobject.h"

int main(int argc, char** argv) {
	TestObj *bar = new TestObj;
	printf("<main>Address of TestObj is %lx\n", (Long_t) bar);
	printf("<main>Performing dynamic_cast of TestObj to TObject, gives us %lx\n", (Long_t) dynamic_cast<TObject*>(bar));
	{
		printf("\n");
		printf("<main>Calling method foo of TestObj...\n");
		bar->foo();
		printf("<main>Back in main...\n");
		printf("<main>Calling method fooVirtual of TestObj...\n");
		bar->fooVirtual();
		printf("<main>Back in main...\n");
		printf("\n");
	}
	{
		printf("<main>Constructing TMethod on foo()...\n");
		TMethod *meth=bar->IsA()->GetMethodAllAny("foo");
		printf("<main>TMethod* is %lx\n", (Long_t) meth);
	        Int_t error;
		printf("<main>Executing prepared method...\n");
	        bar->Execute(meth, NULL, &error);
		printf("<main>Done, error=%d!\n", error);
		printf("<main>Executing method by name...\n");
	        bar->Execute("foo", "", &error);
		printf("<main>Done, error=%d!\n", error);
		printf("\n");
	}
	{
		printf("<main>Constructing TMethod on fooVirtual()...\n");
		TMethod *meth=bar->IsA()->GetMethodAllAny("fooVirtual");
		printf("<main>TMethod* is %lx\n", (Long_t) meth);
	        Int_t error;
		printf("<main>Executing prepared method...\n");
	        bar->Execute(meth, NULL, &error);
		printf("<main>Done, error=%d!\n", error);
		printf("<main>Executing method by name...\n");
	        bar->Execute("fooVirtual", "", &error);
		printf("<main>Done, error=%d!\n", error);
	}
	return 0;
}

