#include "RVersion.h"
#include "TFile.h"

#define R__NESTED_CONTAINER ROOT_VERSION(3,06,2)
#if ROOT_VERSION_CODE < R__NESTED_CONTAINER
#define R__NO_NESTED_CONTAINER
#endif

Bool_t HasNestedContainer(TFile *file) {
   return file->GetVersion() >= R__NESTED_CONTAINER;
}
