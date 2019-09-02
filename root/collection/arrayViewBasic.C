#include "ROOT/RSpan.hxx"

template <typename T>
void show_int_array_it(T view)
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

template <typename T>
void show_int_array_at(T view)
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

template <typename T>
void show_int_array_op(T view)
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

template <typename T>
void show_int_array(T view)
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
    show_int_array<std::span<int>>(good_old_c_array);
    show_int_array<std::span<int>>(array);
    show_int_array<std::span<const int>>(vector);
    show_int_array<std::span<const int>>({1, 2, 3, 4}); //Note initializer_list cannot contain non-const types, but span can.
    show_int_array<std::span<int>>({&good_old_c_array[0], 4});
}
