struct Top {
  int fValue;
};

struct Mid1 : public Top
{
   int fMid1;
};

struct Mid2 : public Top
{
   int fMid2;
};

struct Bottom : public Mid1, Mid2
{
   int fBottom;
};

