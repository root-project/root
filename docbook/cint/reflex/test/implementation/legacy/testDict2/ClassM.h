#ifndef DICT2_CLASSM_H
#define DICT2_CLASSM_H

class ClassM {
public:
   ClassM(): fM('m') {}

   virtual ~ClassM() {}

   int
   m() { return fM; }

   void
   setM(int v) { fM = v; }

private:
   int fM;
};


#endif // DICT2_CLASSM_H
