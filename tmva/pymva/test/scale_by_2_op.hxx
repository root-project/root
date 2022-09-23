#include <vector>
#include <algorithm>
#include <iostream>
namespace Scale_by_2{
void Compute(std::vector<float>& input, std::vector<float>& output){
    std::for_each(input.begin(), input.end(), [](float &i){i *= 2;});
    output = input;
}
} //Double operator