class Simple {
public:
#if VERSION==1
   int fData;
#elif VERSION==2
   float fData;
#else
#error VERSION is not set
#endif
};
