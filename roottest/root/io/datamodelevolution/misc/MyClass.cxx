
#include "MyClass.h"

ClassImp(MyClass)

void MyClass::Print(Option_t* /* option */) const
{
	for (Int_t i = 0; i < Nentries(); ++i)
	{
		if (fArray[i]) 
         fArray[i]->Print();
      else
         printf("gap\n");
	}
}
