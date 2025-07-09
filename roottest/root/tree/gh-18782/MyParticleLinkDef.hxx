#ifdef __CLING__

#pragma link C++ class SplittableBase + ;
#pragma link C++ class std::vector < SplittableBase> + ;

// It is fundamental for the reproducer to have a class with a custom streamer
// and generate the dictionary with the "-" sign
#pragma link C++ class UnsplittableBase - ;
#pragma link C++ class Derived + ;
#pragma link C++ class std::vector < Derived> + ;

#endif
