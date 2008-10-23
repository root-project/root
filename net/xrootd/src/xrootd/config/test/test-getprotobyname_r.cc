#include <netdb.h>
int main(int argc, char **argv)
{
#ifdef __sun
    getprotobyname_r(0, 0, 0, 1);
#else
    getprotobyname_r(0, 0, 0, 1, 0);
#endif
}
