class myiterator {
   int cursor;
public:
   myiterator() : cursor(0) {}

   friend myiterator  operator+(const myiterator&, const myiterator&);
   friend myiterator  operator+(const myiterator&, int);
   friend myiterator &operator++(myiterator&);
   friend myiterator &operator+=(myiterator&, int i);
   friend bool        operator==(const myiterator&, const myiterator&);
   friend bool        operator<(const myiterator&, const myiterator&);
};

myiterator operator+( const myiterator &left , const myiterator &right ) {
   myiterator result = left;
   result.cursor += right.cursor;
   return result;
}

myiterator operator+( const myiterator &left , int right ) {
   myiterator result = left;
   result.cursor += right;
   return result;
}

myiterator &operator++(myiterator &left ) {
   ++left.cursor;
   return left;
}

myiterator& operator+=(myiterator &left, int i ) {
   left.cursor += i;
   return left;
}

bool operator<( const myiterator &left , const myiterator &right ) {
   return left.cursor < right.cursor;
}

bool operator==( const myiterator &left , const myiterator &right ) {
   return left.cursor == right.cursor;
}

bool operator!=( const myiterator &left , const myiterator &right ) {
   return ! (left == right);
}

