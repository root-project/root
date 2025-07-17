#ifndef ROOT_DATAFRAME_TEST_SIMPLE_ELECTRON
#define ROOT_DATAFRAME_TEST_SIMPLE_ELECTRON

// Classes required for the regression test of https://github.com/root-project/root/issues/19392

struct SimpleElectron {
   float electron_pt{};
};

struct Wrapper {
   SimpleElectron electron{};
   float electron_pt{};
};

struct UniqueInner {
   int uniqueInnerInt{};
};

struct UniqueWrapper {
   UniqueInner uniqueInner{};
   int uniqueWrapperInt{};
};

#endif
