namespace pythonizables {

//===========================================================================
class MyBufferReturner {
public:
    MyBufferReturner(int size, double valx, double valy);
    MyBufferReturner(const MyBufferReturner&) = delete;
    MyBufferReturner& operator=(const MyBufferReturner&) = delete;
    ~MyBufferReturner();

public:
    int GetN();
    double* GetX();
    double* GetY();

private:
    double* m_Xbuf;
    double* m_Ybuf;
    int m_size;
};


//===========================================================================
class MyBase {
public:
    virtual ~MyBase();
};
class MyDerived : public MyBase {
public:
    virtual ~MyDerived();
};

MyBase* GimeDerived();

} // namespace pythonizables
