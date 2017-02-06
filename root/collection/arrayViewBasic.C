#include "ROOT/RArrayView.hxx"

void show_int_array_it(std::array_view<int> view)
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

void show_int_array_at(std::array_view<int> view)
{
    std::cout << '{';
    if (!view.empty()) {
        const auto size = view.size();
        for (auto i : ROOT::TSeqI(size)) {
            std::cout << view.at(i);
            if (i!= size-1) {
                std::cout << ", ";
            } else {
                break;
            }
        }
    }
    std::cout << "}\n";
}

void show_int_array_op(std::array_view<int> view)
{
    std::cout << '{';
    if (!view.empty()) {
        const auto size = view.size();
        for (auto i : ROOT::TSeqI(size)) {
            std::cout << view[i];
            if (i!= size-1) {
                std::cout << ", ";
            } else {
                break;
            }
        }
    }
    std::cout << "}\n";
}


void show_int_array(std::array_view<int> view)
{
   show_int_array_at(view);
   show_int_array_op(view);
   show_int_array_it(view);
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
