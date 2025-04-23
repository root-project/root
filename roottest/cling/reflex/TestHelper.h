#include <iostream>

#define RflxAssert(TEST) \
  {if (!(TEST)) std::cout << #TEST << ": FAILED!" << std::endl; \
  else std::cout << #TEST << ": ok." << std::endl;}

#define RflxAssertT(T, TEST) \
  {if (!(TEST)) std::cout << T << " (" << #TEST << "): FAILED!" << std::endl; \
  else std::cout << T << ": ok." << std::endl;}

#define RflxEqual(L, R) \
  {if (!(L == R)) \
     std::cout << #L << " == " << #R << ": FAILED!" << std::endl \
               << "    got \"" << L << "\", expected \"" << R << "\"" << std::endl; \
  else std::cout << #L << " == " << #R << ": ok." << std::endl;}

#define RflxEqualT(T, L, R) \
  {if (!(L == R)) \
     std::cout << T << " (" << #L << " == " << #R << "): FAILED!" << std::endl \
               << "    got \"" << L << "\", expected \"" << R << "\"" << std::endl; \
  else std::cout << T << ": ok." << std::endl;}
