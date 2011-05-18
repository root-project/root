#include <sys/socket.h>
#include <netdb.h>
int main(int argc, char **argv)
{
   getnameinfo(0, 1, 0, 1, 0, 1, 1);
}
