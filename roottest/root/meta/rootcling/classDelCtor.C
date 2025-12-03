class TheClassWithDelCtor{
public:
   TheClassWithDelCtor() = delete;
   TheClassWithDelCtor(int) {}
};

void classDelCtor()
{
   TheClassWithDelCtor a(0);
}
