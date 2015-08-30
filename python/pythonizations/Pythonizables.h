//===========================================================================
class MyBufferReturner {
public:
    MyBufferReturner(int size);
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
