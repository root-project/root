int get_histogram(char const* , char const* ){return 0; }
template<class T> void get_histogram(T const & t, char const * name);
#ifdef __CINT__
#pragma link C++ function get_histogram;
#endif

void constissue() {}
