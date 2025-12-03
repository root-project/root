#ifndef CLASSWITHNESTEDSAMENAME
#define CLASSWITHNESTEDSAMENAME

struct Particle {
   float fPt{42.f};
};

struct DataMember {
   Particle fInner{};
};

struct TopLevel {
   DataMember fInner{};
};

#endif // CLASSWITHNESTEDSAMENAME
