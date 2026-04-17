#include "civetweb.h"

int main()
{
    printf("%d", mg_check_feature(0xFFF));
    return 0;
}
