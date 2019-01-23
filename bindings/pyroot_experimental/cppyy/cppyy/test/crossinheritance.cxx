#include "crossinheritance.h"


// for overridden method checking
CrossInheritance::Base1::~Base1() {}

int CrossInheritance::Base1::call_get_value(Base1* b)
{
    return b->get_value();
}

int CrossInheritance::Base1::call_sum_value(Base1* b, int i)
{
    return b->sum_value(i);
}

int CrossInheritance::Base1::call_sum_all(Base1* b, int i)
{
    return b->sum_all(i);
}

int CrossInheritance::Base1::call_sum_all(Base1* b, int i, int j)
{
    return b->sum_all(i, j);
}
