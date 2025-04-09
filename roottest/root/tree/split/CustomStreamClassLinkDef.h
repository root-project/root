#ifdef __ROOTCLING__

#ifdef CUSTOM_STREAMER
#pragma link C++ class MyClass-;
#else
#pragma link C++ class MyClass+;
#endif 

#endif
