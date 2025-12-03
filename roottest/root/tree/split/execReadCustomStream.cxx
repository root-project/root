#define CUSTOM_STREAMER

#include "CustomStreamClass.cpp"
#include "CustomStreamClassLinkDef.h"
#include "CustomStreamClassFunc.cpp"

int execReadCustomStream() {
   return ReadTree();
}