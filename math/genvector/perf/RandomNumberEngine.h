
#include <random>

static std::default_random_engine ggen;

static std::uniform_real_distribution<double> p_x(1, 2), p_y(2, 3), p_z(3, 4);
static std::uniform_real_distribution<double> d_x(1, 2), d_y(2, 3), d_z(3, 4);
static std::uniform_real_distribution<double> c_x(1, 2), c_y(2, 3), c_z(3, 4);
static std::uniform_real_distribution<double> p0(-0.002, 0.002), p1(-0.2, 0.2), p2(0.97, 0.99), p3(-1300, 1300);
