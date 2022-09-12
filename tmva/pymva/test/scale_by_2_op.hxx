#include <vector>
#include <algorithm>
#include <iostream>
namespace Scale_by_2{
std::vector<std::vector<float>> Compute(std::vector<std::vector<float>> inputs){
    for(auto& it:inputs){
        std::for_each(it.begin(), it.end(), [](float &i){i *= 2;});
    }
    return inputs;
}
} //Double operator