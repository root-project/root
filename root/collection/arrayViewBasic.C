#include "ROOT/RArrayView.hxx"

void show_int_array(std::array_view<int> view)
{
    std::cout << '{';
    if (!view.empty()) {
        auto itr = view.begin();
        auto const end = view.end();
        while (true) {
            std::cout << *itr;
            if (++itr != end) {
                std::cout << ", ";
            } else {
                break;
            }
        }
    }
    std::cout << "}\n";
}

void arrayViewBasic()
{
    int good_old_c_array[] = {1, 2, 3, 4};
    std::array<int, 4> array  {{1, 2, 3, 4}};
    std::vector<int> vector  {1, 2, 3, 4};
    std::vector<float> vectorf {1.f, 2.f, 3.f, 4.f};

    // access arrays with safe and unified way
    show_int_array(good_old_c_array);
    show_int_array(array);
    show_int_array(vector);
    show_int_array({1, 2, 3, 4});
    show_int_array({&good_old_c_array[0], 4});
}
