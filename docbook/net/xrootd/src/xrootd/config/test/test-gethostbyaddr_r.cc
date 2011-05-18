#include <netdb.h>
int main(int argc, char **argv)
{
#ifdef __sun
    gethostbyaddr_r(0, 1, 1, 0, 0, 1, 0);
#else
    gethostbyaddr_r(0, 1, 1, 0, 0, 1, 0, 0);
#endif
}
