#ifndef CORAL_BLOB_H
#define CORAL_BLOB_H 1

namespace coral
{

  /**
   * @class Blob Blob.h CoralBase/Blob.h
   *
   * A class defining a BLOB type
   */
  class Blob
  {

  public:

    /// Default Constructor. Creates an empty BLOB
    Blob();

    /// Constructor initializing a BLOB with initialSize bytes
    explicit Blob( long initialSizeInBytes );

    /// Destructor. Frees internally allocated memory
    ~Blob();

    /// Copy constructor
    Blob( const Blob& rhs );

    /// Assignment operator
    Blob& operator=( const Blob& rhs );

    /// Appends the data of another blob
    Blob& operator+=( const Blob& rhs );

    /// Equal operator. Compares the contents of the binary blocks
    bool operator==( const Blob& rhs ) const;

    /// Comparison operator
    bool operator!=( const Blob& rhs ) const;

    /// Returns the starting address of the BLOB
    const void* startingAddress() const;

    /// Returns the starting address of the BLOB
    void* startingAddress();

    /// Current size of the blob
    long size() const;

    /// Extends the BLOB by additionalSizeInBytes
    void extend( long additionalSizeInBytes );

    /// Resizes a BLOB to sizeInBytes
    void resize( long sizeInBytes );

  private:

    /// The current size of the BLOB
    long m_size;

    /// The BLOB data buffer
    void* m_data;

  };

}

// Inline methods
inline bool
coral::Blob::operator!=( const Blob& rhs ) const
{
  return ( ! ( this->operator==( rhs ) ) );
}

#endif
