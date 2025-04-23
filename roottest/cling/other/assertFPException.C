#pragma clang diagnostic ignored "-Wdivision-by-zero"

// ROOT-8080: do not crash.
{
int i = 8/0;
auto nastylambda = [](int a){return a/0;};
nastylambda(1);
0;
}
