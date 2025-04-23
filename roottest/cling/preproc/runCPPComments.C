#define SOMETHING

# if /*
blah
    */0
class test {
public:
   int b();
   
};
# elif /* another 
comment*/ defined(SOMETHING)    /*yet another one*/
    class test 
    {
    public:
       int a() { return printf("a\n");}
    };


#endif

void runCPPComments() 
{
   test t;
   t.a();
}
