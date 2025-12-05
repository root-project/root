#ifndef ROOT_DATAFRAME_TEST_SIMPLE_ELECTRON
#define ROOT_DATAFRAME_TEST_SIMPLE_ELECTRON

struct SimpleElectron {
   float electron_pt{};
};

struct Wrapper {
   SimpleElectron electron{};
   float electron_pt{};
};

#endif
