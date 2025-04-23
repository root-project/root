// BNoClassDef.h

#ifndef BNOCLASSDEF_HDR
#define BNOCLASSDEF_HDR

class B {
private:
  char*         fName;
  char*         fTitle;
  int           fX;
  double        fY;
public:
  B();
  B(char const *name, char const *title, int x, double y);
  B(const B&);
  B& operator=(const B&);
  ~B();
public:
  char const*   GetName() const { return fName; }
  char const*   GetTitle() const { return fTitle; }
  int           GetX() const     { return fX; }
  double        GetY() const     { return fY; }
  void          SetName(char const*);
  void          SetTitle(char const*);
  void          SetX(int val)    { fX = val; }
  void          SetY(double val) { fY = val; }
public:
  void          repr() const;
};

#endif // BNOCLASSDEF_HDR
