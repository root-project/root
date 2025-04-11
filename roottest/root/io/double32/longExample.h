#include "TObject.h"

struct m01 {
  Double32_t fValue[3];
  Double32_t fV2[4];
};

template <typename V> struct alloc {};
template <typename V> struct vec {};

template <typename K, typename V> struct map1 {};
template <typename K, typename V, typename A = alloc<pair<K,V> > > struct map2 {};


template <typename T> class m02 : map2<T, T> {
public:
   m02() : ff4(0),fv3(fv2),fv4(0),fv5(5),fv6(0) {}

   int fN;

   vector<Double32_t> ff1;
   Double32_t ff2[3];
   Double32_t ff3;
   Double32_t *ff4; //[fN]

   T fv1[4];
   T fv2;
   T& fv3;
   T* fv4; //[fN]
   const T fv5;
   const T* fv6; //[fN]
   vec<T> fv7;
   map1<Int_t,T> fv9;
   map1<Int_t,vec<T> > fv10;
   map1<Int_t,map1<double,T> > fv11;
   map1<Int_t,map1<double,vec<T> > > fv12;

   map2<Int_t,T> fv13;
   map2<Int_t,vec<T> > fv14;
   map2<Int_t,map2<double,T> > fv15;
   map2<Int_t,map2<double,vec<T> > > fv16;

};

#ifdef __ROOTCLING__
#pragma link C++ class m01+;
#pragma link C++ class m02<Double32_t>+;

#pragma link C++ class map2<Double32_t,Double32_t>+;

#pragma link C++ class vec<Double32_t>+;

#pragma link C++ class map1<Int_t,Double32_t>+;
#pragma link C++ class map1<Int_t,vec<Double32_t> >+;
#pragma link C++ class map1<Int_t,map1<double,Double32_t> >+;
#pragma link C++ class map1<Int_t,map1<double,vec<Double32_t> > >+;

#pragma link C++ class map2<Int_t,Double32_t>+;
#pragma link C++ class map2<Int_t,vec<Double32_t> >+;
#pragma link C++ class map2<Int_t,map2<double,Double32_t> >+;
#pragma link C++ class map2<Int_t,map2<double,vec<Double32_t> > >+;
#endif

#if 0

This is what clang sees:

.class m02<Double32_t>
===========================================================================
class m02<double>
SIZE: 160 FILE: test01.cxx LINE: 15
Base classes: -------------------------------------------------------------
0x0        private map2<double, double, struct alloc<struct std::__1::pair<double, double> > >
List of member variables: -------------------------------------------------
test01.cxx       17 0x0        public: int fN
test01.cxx       19 0x8        public: vector<Double32_t> ff1, size = 24
test01.cxx       20 0x20       public: Double32_t ff2[3]
test01.cxx       21 0x38       public: Double32_t ff3
test01.cxx       22 0x40       public: Double32_t *ff4
test01.cxx       24 0x48       public: double fv1[4]
test01.cxx       25 0x68       public: double fv2
test01.cxx       26 0x70       public: double &fv3
test01.cxx       27 0x78       public: double *fv4
test01.cxx       28 0x80       public: const double fv5
test01.cxx       29 0x88       public: const double *fv6
test01.cxx       30 0x90       public: vec<double> fv7, size = 1
test01.cxx       31 0x91       public: map1<Int_t, double> fv9, size = 1
test01.cxx       32 0x92       public: map1<Int_t, vec<double> > fv10, size = 1
test01.cxx       33 0x93       public: map1<Int_t, map1<double, double> > fv11, size = 1
test01.cxx       34 0x94       public: map1<Int_t, map1<double, vec<double> > > fv12, size = 1
test01.cxx       36 0x95       public: map2<Int_t, double> fv13, size = 1
test01.cxx       37 0x96       public: map2<Int_t, vec<double> > fv14, size = 1
test01.cxx       38 0x97       public: map2<Int_t, map2<double, double> > fv15, size = 1
test01.cxx       39 0x98       public: map2<Int_t, map2<double, vec<double> > > fv16, size = 1
List of member functions: -------------------------------------------------
filename     line:size busy function type and name

and this is what StreamerInfo need to see:

StreamerInfo for class: m02<Double32_t>, checksum=0x8610c861
map2<Double32_t,Double32_t,alloc<pair<Double32_t,Double32_t> > > BASE            offset=  0 type= 0
int            fN              offset=  0 type= 6
vector<Double32_t> ff1             offset=  8 type=300 ,stl=1, ctype=9,
Double32_t     ff2[3]          offset= 32 type=29
Double32_t     ff3             offset= 56 type= 9
Double32_t*    ff4             offset= 64 type=49 [fN]
Double32_t     fv1[4]          offset= 72 type=29
Double32_t     fv2             offset=104 type= 9
Double32_t&    fv3             offset=112 type= 9
Double32_t*    fv4             offset=120 type=49 [fN]
Double32_t     fv5             offset=128 type= 9
Double32_t*    fv6             offset=136 type=49 [fN]
vec<Double32_t> fv7             offset=144 type=62
map1<int,Double32_t> fv9             offset=145 type=62
map1<int,vec<Double32_t> > fv10            offset=146 type=62
map1<int,map1<double,Double32_t> > fv11            offset=147 type=62
map1<int,map1<double,vec<Double32_t> > > fv12            offset=148 type=62
map2<int,Double32_t,alloc<pair<int,Double32_t> > > fv13            offset=149 type=62
map2<int,vec<Double32_t>,alloc<pair<int,vec<Double32_t> > > > fv14            offset=150 type=62
map2<int,map2<double,Double32_t,alloc<pair<double,Double32_t> > >,alloc<pair<int,map2<double,Double32_t,alloc<pair<double,Double32_t> > > > > > fv15            offset=151 type=62
map2<int,map2<double,vec<Double32_t>,alloc<pair<double,vec<Double32_t> > > >,alloc<pair<int,map2<double,vec<Double32_t>,alloc<pair<double,vec<Double32_t> > > > > > > fv16            offset=152 type=62
i= 0, map2<Double32_t,Double32_t,alloc<pair<Double32_t,Double32_t> > > type=  0, offset=  0, len=1, method=0
i= 1, fN              type=  6, offset=  0, len=1, method=140226211206296
i= 2, ff1             type=300, offset=  8, len=1, method=0
i= 3, ff2             type= 29, offset= 32, len=4, method=0
i= 4, ff4             type= 49, offset= 64, len=1, method=0
i= 5, fv1             type= 29, offset= 72, len=6, method=0
i= 6, fv4             type= 29, offset=120, len=2, method=0
i= 7, fv6             type= 49, offset=136, len=1, method=0
i= 8, fv7             type= 62, offset=144, len=1, method=0
i= 9, fv9             type= 62, offset=145, len=1, method=0
i=10, fv10            type= 62, offset=146, len=1, method=0
i=11, fv11            type= 62, offset=147, len=1, method=0
i=12, fv12            type= 62, offset=148, len=1, method=0
i=13, fv13            type= 62, offset=149, len=1, method=0
i=14, fv14            type= 62, offset=150, len=1, method=0
i=15, fv15            type= 62, offset=151, len=1, method=0
i=16, fv16            type= 62, offset=152, len=1, method=0

#endif

