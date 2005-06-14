class A {
public:
   A() { m_a = 1; }
   virtual ~A() {}
   virtual int GetValue() = 0;

public:
   int m_a;
};

class B : public virtual A {
public:
   B() { m_b = 2; }
   virtual int GetValue() { return m_b; }

public:
   int m_b;
};

// NOTE: class C : public virtual A, public virtual B
//  causes gROOT->GetClass() to fail ...
class C : public virtual B, public virtual A {
public:
   C() { m_c = 3; }
   virtual int GetValue() { return m_c; }

public:
   int m_c;
};

class D : public virtual C, public virtual A {
public:
   D() { m_d = 4; }
   virtual int GetValue() { return m_d; }

public:
   int m_d;
};

int GetA( A& a ) { return a.m_a; }
int GetB( B& b ) { return b.m_b; }
int GetC( C& c ) { return c.m_c; }
int GetD( D& d ) { return d.m_d; }
