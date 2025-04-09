R__LOAD_LIBRARY(libcmswrapper_dictrflx)

template <class X>
struct cmsWrapperTest {
   typedef edm::Wrapper<double> wd;
};

void execcmsWrapper() {
   cmsWrapperTest<int> wti;
}
