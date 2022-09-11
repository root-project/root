#include <vector>
#include <algorithm>
#include <iostream>
namespace Double{
std::vector<std::vector<float>> Compute(std::vector<std::vector<float>> inputs){
    for(auto&it: inputs){
        for(auto& i:it){
            std::cout<<i<<"\n";
        }
    }
    for(auto& it:inputs){
        std::for_each(it.begin(), it.end(), [](float &i){i *= 2;});
    }
    return inputs;
}
} //Double operator