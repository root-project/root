{
// Declare a function
gInterpreter->Declare("float func(float x) { return 2.0 * x; }");

// Get the pointer to the function
auto ptr = gInterpreter->Calc("&func");
//float (*func)(float) = reinterpret_cast<float(*)(float)>(ptr);

// Call the function
std::cout << func(42.0) << std::endl;
}
