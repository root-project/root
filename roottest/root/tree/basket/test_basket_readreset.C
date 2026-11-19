#include "TBasket.h"
#include "TBranch.h"
#include "TBufferFile.h"
#include <iostream>

// Subclass to bypass the protected visibility of ReadResetBuffer
class TestBasket : public TBasket {
public:
    TestBasket(TBranch *branch) {
        fBranch = branch; 
    }
    
    void InvokeReadResetBuffer(Int_t basketnumber) {
        ReadResetBuffer(basketnumber);
    }
    
    void SetBufferForTesting(TBuffer *buf) {
        fBuffer = buf;
    }
};

void testTBasketReset() {
    // Initialize mock environment
    TBranch dummyBranch;
    Int_t totalBaskets = 15;
    dummyBranch.SetBufferEntries(totalBaskets); 
    
    Int_t currentBasketIdx = 2;
    Int_t maximumNeededInWindow = 2000; 
    
    // Setup lookahead window metadata for the next 10 baskets
    for (int i = 0; i < totalBaskets; ++i) {
        if (i >= currentBasketIdx && i <= (currentBasketIdx + 10)) {
            dummyBranch.GetBasketBytes()[i] = maximumNeededInWindow;
        } else {
            dummyBranch.GetBasketBytes()[i] = 1000;
        }
    }

    TestBasket basket(&dummyBranch);
    
    // Allocate heavily bloated buffer 
    Int_t massiveBufferSize = 500000; 
    TBufferFile* bloatedBuffer = new TBufferFile(TBuffer::kRead, massiveBufferSize);
    bloatedBuffer->SetBufferOffset(massiveBufferSize);
    basket.SetBufferForTesting(bloatedBuffer);

    std::cout << "Initial buffer size: " << bloatedBuffer->BufferSize() << std::endl;

    // Trigger the shrinking behavior
    basket.InvokeReadResetBuffer(currentBasketIdx);

    Int_t postResetSize = bloatedBuffer->BufferSize();
    std::cout << "Post-reset buffer size is smaller than initial: " 
              << (postResetSize < massiveBufferSize ? "YES" : "NO") << std::endl;
              
    std::cout << "Post-reset buffer satisfies safety floor requirement: " 
              << (postResetSize >= maximumNeededInWindow ? "YES" : "NO") << std::endl;

    delete bloatedBuffer;
}
