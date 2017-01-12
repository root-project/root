#ifndef PAR_H
#define PAR_H
//custom class for tests
template<class T> class Particle {
   T x;
   T y;
public:
   Particle(T _x = 0, T _y = 0): x(_x), y(_y) {}
   void Set(T _x, T _y)
   {
      x = _x;
      y = _y;
   }
   T GetX()
   {
      return x;
   }
   T GetY()
   {
      return y;
   }

   void Print()
   {
      std::cout << "x = " << x << " y = " << y << std::endl;
      std::cout.flush();
   }
};
#endif
