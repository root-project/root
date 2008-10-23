#define _GNU_SOURCE
#include <unistd.h>
int main(int argc, char **argv)
{
   setresuid(0, 0, 0);
   setresgid(0, 0, 0);
}
