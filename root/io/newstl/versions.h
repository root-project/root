#include "RVersion.h"
#include "TFile.h"
#include "TestOutput.h"

#define R__NESTED_CONTAINER ROOT_VERSION(3,12,0)
#if ROOT_VERSION_CODE < R__NESTED_CONTAINER
#define R__NO_NESTED_CONTAINER
#endif

#define R__VAR_ARRAY_OF_CONTAINERS ROOT_VERSION(3,12,0)
#if ROOT_VERSION_CODE < R__VAR_ARRAY_OF_CONTAINERS
#define R__NO_VAR_ARRAY_OF_CONTAINERS
#endif

#define R__CAN_SPLIT_STL_CONTAINER ROOT_VERSION(3,12,0)
#if ROOT_VERSION_CODE < R__CAN_SPLIT_STL_CONTAINER
#define R__CANNOT_SPLIT_STL_CONTAINER
#endif

#define R__NESTED_CONST_STRING ROOT_VERSION(3,12,0)
#if ROOT_VERSION_CODE < R__NESTED_CONST_STRING
#define R__NO_NESTED_CONST_STRING
#endif

class OutputOneErrorMessage {
 public:
   OutputOneErrorMessage(const std::string &msg) {
      Unsupported(msg);
   }
};

Bool_t HasNestedContainer(TFile *file) {
   Bool_t result = file->GetVersion() >= R__NESTED_CONTAINER;
   if (!result) {
      static OutputOneErrorMessage error("nested stl containers");
   }
   return result;
}

Bool_t HasNestedConstString(TFile *file) {
   Bool_t result = file->GetVersion() >= R__NESTED_CONST_STRING;
   if (!result) {
      static OutputOneErrorMessage error("stl containers of const string*");
   }
   return result;
}

bool HasSplitStlContainer(TFile *file, int splitlevel) {
   Bool_t result = (splitlevel<2) || (file->GetVersion() >= R__NESTED_CONTAINER);
   if (!result) {
      static OutputOneErrorMessage error("splitting of stl containers");
   }
   return result;
}

bool HasVarArrayOfContainers(TFile *file) {
   Bool_t result = file->GetVersion() >= R__VAR_ARRAY_OF_CONTAINERS;
   if (!result) {
      static OutputOneErrorMessage error("variable size array of stl containers");
   }
   return result;
}

