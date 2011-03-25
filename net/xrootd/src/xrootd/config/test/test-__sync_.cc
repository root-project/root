int main()
{
    unsigned long val = 111, tmp, *mem = &val;

    if (__sync_fetch_and_add(&val, 111) != 111 || val != 222) return 1;
    if (__sync_add_and_fetch(&val, 111) != 333)               return 1;
    if (__sync_sub_and_fetch(&val, 111) != 222)               return 1;
    if (__sync_fetch_and_sub(&val, 111) != 222 || val != 111) return 1;

    if (__sync_fetch_and_or (&val, 0)   != 111 || val != 111) return 1;
    if (__sync_fetch_and_and(&val, 0)   != 111 || val != 0  ) return 1;

    if (__sync_bool_compare_and_swap(mem, 0, 444) == 0 || val != 444)
        return 1;

    return 0;
}
