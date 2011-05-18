#include <fcntl.h>
#include <sys/stat.h>
int main(int argc, char **argv)
{
   int rc= 0;
   struct stat st;
   if (fstatat(0, "/tmp", &st, 0) != 0) rc = 1;
}
