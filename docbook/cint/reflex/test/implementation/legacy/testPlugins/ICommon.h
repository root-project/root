#ifndef ICOMMON_H
#define ICOMMON_H

struct ICommon {
   virtual int do_nothing(int) = 0;
   virtual std::string do_something() = 0;
   virtual double get_f() = 0;
   virtual ~ICommon() {}

};

class Base {
public:
   Base() {}

   virtual ~Base() {}

   virtual double
   do_base(double d) { return d; }

private:
   int data[10];
};

struct ID {
   ID(int a, int b): m_a(a),
      m_b(b) {}

   bool
   operator ==(const ID& id) { return m_a == id.m_a && m_b == id.m_b; }

   int m_a;
   int m_b;

};


inline std::ostream&
operator <<(std::ostream& s,
            const ID& id) {
   s << "ID_" << id.m_a << "_" << id.m_b;
   return s;
}


#endif
