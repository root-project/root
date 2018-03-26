#include <../src/BidirMMapPipe.h>

#include <sys/time.h>
#include <iomanip>

#include "gtest/gtest.h"

using namespace RooFit;

int simplechild(BidirMMapPipe& pipe)
{
    // child does an echo loop
    while (pipe.good() && !pipe.eof()) {
        // read a string
        std::string str;
        pipe >> str;
        if (!pipe) return -1;
        if (pipe.eof()) break;
        if (!str.empty()) {
            std::cout << "[CHILD] :  read: " << str << std::endl;
            str = "... early in the morning?";
        }
        pipe << str << BidirMMapPipe::flush;
        // did our parent tell us to shut down?
        if (str.empty()) break;
        if (!pipe) return -1;
        if (pipe.eof()) break;
        std::cout << "[CHILD] : wrote: " << str << std::endl;
    }
    pipe.close();
    return 0;
}

#include <sstream>
int randomchild(BidirMMapPipe& pipe)
{
    // child sends out something at random intervals
    ::srand48(::getpid());
    {
        // wait for parent's go ahead signal
        std::string s;
        pipe >> s;
    }
    // no shutdown sequence needed on this side - we're producing the data,
    // and the parent can just read until we're done (when it'll get EOF)
    for (int i = 0; i < 5; ++i) {
        // sleep a random time between 0 and .9 seconds
        ::usleep(int(1e6 * ::drand48()));
        std::ostringstream buf;
        buf << "child pid " << ::getpid() << " sends message " << i;
        std::string str = buf.str();
        std::cout << "[CHILD] : " << str << std::endl;
        pipe << str << BidirMMapPipe::flush;
        if (!pipe) return -1;
        if (pipe.eof()) break;
    }
    // tell parent we're shutting down
    pipe << "" << BidirMMapPipe::flush;
    // wait for parent to acknowledge
    std::string s;
    pipe >> s;
    pipe.close();
    return 0;
}

int benchchildrtt(BidirMMapPipe& pipe)
{
    // child does the equivalent of listening for pings and sending the
    // packet back
    char* str = 0;
    while (pipe && !pipe.eof()) {
        pipe >> str;
        if (!pipe) {
            std::free(str);
            pipe.close();
            return -1;
        }
        if (pipe.eof()) break;
        pipe << str << BidirMMapPipe::flush;
        // if we have just completed the shutdown handshake, we break here
        if (!std::strlen(str)) break;
    }
    std::free(str);
    pipe.close();
    return 0;
}

int benchchildsink(BidirMMapPipe& pipe)
{
    // child behaves like a sink
    char* str = 0;
    while (pipe && !pipe.eof()) {
        pipe >> str;
        if (!std::strlen(str)) break;
    }
    pipe << "" << BidirMMapPipe::flush;
    std::free(str);
    pipe.close();
    return 0;
}

int benchchildsource(BidirMMapPipe& pipe)
{
    // child behaves like a source
    char* str = 0;
    for (unsigned i = 0; i <= 24; ++i) {
        str = reinterpret_cast<char*>(std::realloc(str, (1 << i) + 1));
        std::memset(str, '4', 1 << i);
        str[1 << i] = 0;
        for (unsigned j = 0; j < 1 << 7; ++j) {
            pipe << str;
            if (!pipe || pipe.eof()) {
                std::free(str);
                pipe.close();
                return -1;
            }
        }
        // tell parent we're done with this block size
        pipe << "" << BidirMMapPipe::flush;
    }
    // tell parent to shut down
    pipe << "" << BidirMMapPipe::flush;
    std::free(str);
    pipe.close();
    return 0;
}

BidirMMapPipe* spawnChild(int (*childexec)(BidirMMapPipe&))
{
    // create a pipe with the given child at the remote end
    BidirMMapPipe *p = new BidirMMapPipe();
    if (p->isChild()) {
        int retVal = childexec(*p);
        delete p;
        std::exit(retVal);
    }
    return p;
}

// simple echo loop test
TEST(testBidirMMAPPipe_simple, getResult){
    std::cout << "[PARENT]: simple challenge-response test, "
        "one child:" << std::endl;
    BidirMMapPipe* pipe = spawnChild(simplechild);
    for (int i = 0; i < 5; ++i) {
        std::string str("What shall we do with a drunken sailor...");
        *pipe << str << BidirMMapPipe::flush;
        ASSERT_TRUE(*pipe);
        std::cout << "[PARENT]: wrote: " << str << std::endl;
        *pipe >> str;
        ASSERT_TRUE(*pipe);
        std::cout << "[PARENT]:  read: " << str << std::endl;
    }
    // send shutdown string
    *pipe << "" << BidirMMapPipe::flush;
    // wait for shutdown handshake
    std::string s;
    *pipe >> s;
    int retVal = pipe->close();
    std::cout << "[PARENT]: exit status of child: " << retVal <<
        std::endl;
    delete pipe;
}

    // simple poll test - children send 5 results in random intervals
TEST(testBidirMMAPPipe_poll, getResult)
    {
        unsigned nch = 20;
        std::cout << std::endl << "[PARENT]: polling test, " << nch <<
            " children:" << std::endl;
        typedef BidirMMapPipe::PollEntry PollEntry;
        // poll data structure
        BidirMMapPipe::PollVector pipes;
        pipes.reserve(nch);
        // spawn children
        for (unsigned i = 0; i < nch; ++i) {
            std::cout << "[PARENT]: spawning child " << i << std::endl;
            pipes.push_back(PollEntry(spawnChild(randomchild),
                        BidirMMapPipe::Readable));
        }
        // wake children up
        std::cout << "[PARENT]: waking up children" << std::endl;
        for (unsigned i = 0; i < nch; ++i)
            *pipes[i].pipe << "" << BidirMMapPipe::flush;
        std::cout << "[PARENT]: waiting for events on children's pipes" << std::endl;
        // while at least some children alive
        while (!pipes.empty()) {
            // poll, wait until status change (infinite timeout)
            int npipes = BidirMMapPipe::poll(pipes, -1);
            // scan for pipes with changed status
            for (std::vector<PollEntry>::iterator it = pipes.begin();
                    npipes && pipes.end() != it; ) {
                if (!it->revents) {
                    // unchanged, next one
                    ++it;
                    continue;
                }
                --npipes; // maybe we can stop early...
                // read from pipes which are readable
                if (it->revents & BidirMMapPipe::Readable) {
                    std::string s;
                    *(it->pipe) >> s;
                    if (!s.empty()) {
                        std::cout << "[PARENT]: Read from pipe " << it->pipe <<
                            ": " << s << std::endl;
                        ++it;
                        continue;
                    } else {
                        // child is shutting down...
                        *(it->pipe) << "" << BidirMMapPipe::flush;
                        goto childcloses;
                    }
                }
                // retire pipes with error or end-of-file condition
                if (it->revents & (BidirMMapPipe::Error |
                            BidirMMapPipe::EndOfFile |
                            BidirMMapPipe::Invalid)) {
                    std::cerr << "[DEBUG]: Event on pipe " << it->pipe <<
                        " revents" <<
                        ((it->revents & BidirMMapPipe::Readable) ? " Readable" : "") <<
                        ((it->revents & BidirMMapPipe::Writable) ? " Writable" : "") <<
                        ((it->revents & BidirMMapPipe::ReadError) ? " ReadError" : "") <<
                        ((it->revents & BidirMMapPipe::WriteError) ? " WriteError" : "") <<
                        ((it->revents & BidirMMapPipe::ReadEndOfFile) ? " ReadEndOfFile" : "") <<
                        ((it->revents & BidirMMapPipe::WriteEndOfFile) ? " WriteEndOfFile" : "") <<
                        ((it->revents & BidirMMapPipe::ReadInvalid) ? " ReadInvalid" : "") <<
                        ((it->revents & BidirMMapPipe::WriteInvalid) ? " WriteInvalid" : "") <<
                        std::endl;
childcloses:
                    int retVal = it->pipe->close();
                    std::cout << "[PARENT]: child exit status: " <<
                        retVal << ", number of children still alive: " <<
                        (pipes.size() - 1) << std::endl;
                    delete it->pipe;
                    it = pipes.erase(it);
                    continue;
                }
            }
        }
    }
    //~ // little benchmark - round trip time
    //~ {
        //~ std::cout << std::endl << "[PARENT]: benchmark: round-trip times vs block size" << std::endl;
        //~ for (unsigned i = 0; i <= 24; ++i) {
            //~ char *s = new char[1 + (1 << i)];
            //~ std::memset(s, 'A', 1 << i);
            //~ s[1 << i] = 0;
            //~ const unsigned n = 1 << 7;
            //~ double avg = 0., min = 1e42, max = -1e42;
            //~ BidirMMapPipe *pipe = spawnChild(benchchildrtt);
            //~ for (unsigned j = n; j--; ) {
                //~ struct timeval t1;
                //~ ::gettimeofday(&t1, 0);
                //~ *pipe << s << BidirMMapPipe::flush;
                //~ if (!*pipe || pipe->eof()) break;
                //~ *pipe >> s;
                //~ if (!*pipe || pipe->eof()) break;
                //~ struct timeval t2;
                //~ ::gettimeofday(&t2, 0);
                //~ t2.tv_sec -= t1.tv_sec;
                //~ t2.tv_usec -= t1.tv_usec;
                //~ double dt = 1e-6 * double(t2.tv_usec) + double(t2.tv_sec);
                //~ if (dt < min) min = dt;
                //~ if (dt > max) max = dt;
                //~ avg += dt;
            //~ }
            //~ // send a shutdown string
            //~ *pipe << "" << BidirMMapPipe::flush;
            //~ // get child's shutdown ok
            //~ *pipe >> s;
            //~ avg /= double(n);
            //~ avg *= 1e6; min *= 1e6; max *= 1e6;
            //~ int retVal = pipe->close();
            //~ if (retVal) {
                //~ std::cout << "[PARENT]: child exited with code " << retVal << std::endl;
                //~ return retVal;
            //~ }
            //~ delete pipe;
            //~ // there is a factor 2 in the formula for the transfer rate below,
            //~ // because we transfer data of twice the size of the block - once
            //~ // to the child, and once for the return trip
            //~ std::cout << "block size " << std::setw(9) << (1 << i) <<
                //~ " avg " << std::setw(7) << avg << " us min " <<
                //~ std::setw(7) << min << " us max " << std::setw(7) << max <<
                //~ "us speed " << std::setw(9) <<
                //~ 2. * (double(1 << i) / double(1 << 20) / (1e-6 * avg)) <<
                //~ " MB/s" << std::endl;
            //~ delete[] s;
        //~ }
        //~ std::cout << "[PARENT]: all children had exit code 0" << std::endl;
    //~ }
    //~ // little benchmark - child as sink
    //~ {
        //~ std::cout << std::endl << "[PARENT]: benchmark: raw transfer rate with child as sink" << std::endl;
        //~ for (unsigned i = 0; i <= 24; ++i) {
            //~ char *s = new char[1 + (1 << i)];
            //~ std::memset(s, 'A', 1 << i);
            //~ s[1 << i] = 0;
            //~ const unsigned n = 1 << 7;
            //~ double avg = 0., min = 1e42, max = -1e42;
            //~ BidirMMapPipe *pipe = spawnChild(benchchildsink);
            //~ for (unsigned j = n; j--; ) {
                //~ struct timeval t1;
                //~ ::gettimeofday(&t1, 0);
                //~ // streaming mode - we do not flush here
                //~ *pipe << s;
                //~ if (!*pipe || pipe->eof()) break;
                //~ struct timeval t2;
                //~ ::gettimeofday(&t2, 0);
                //~ t2.tv_sec -= t1.tv_sec;
                //~ t2.tv_usec -= t1.tv_usec;
                //~ double dt = 1e-6 * double(t2.tv_usec) + double(t2.tv_sec);
                //~ if (dt < min) min = dt;
                //~ if (dt > max) max = dt;
                //~ avg += dt;
            //~ }
            //~ // send a shutdown string
            //~ *pipe << "" << BidirMMapPipe::flush;
            //~ // get child's shutdown ok
            //~ *pipe >> s;
            //~ avg /= double(n);
            //~ avg *= 1e6; min *= 1e6; max *= 1e6;
            //~ int retVal = pipe->close();
            //~ if (retVal) {
                //~ std::cout << "[PARENT]: child exited with code " << retVal << std::endl;
                //~ return retVal;
            //~ }
            //~ delete pipe;
            //~ std::cout << "block size " << std::setw(9) << (1 << i) <<
                //~ " avg " << std::setw(7) << avg << " us min " <<
                //~ std::setw(7) << min << " us max " << std::setw(7) << max <<
                //~ "us speed " << std::setw(9) <<
                //~ (double(1 << i) / double(1 << 20) / (1e-6 * avg)) <<
                //~ " MB/s" << std::endl;
            //~ delete[] s;
        //~ }
        //~ std::cout << "[PARENT]: all children had exit code 0" << std::endl;
    //~ }
    //~ // little benchmark - child as source
    //~ {
        //~ std::cout << std::endl << "[PARENT]: benchmark: raw transfer rate with child as source" << std::endl;
        //~ char *s = 0;
        //~ double avg = 0., min = 1e42, max = -1e42;
        //~ unsigned n = 0, bsz = 0;
        //~ BidirMMapPipe *pipe = spawnChild(benchchildsource);
        //~ while (*pipe && !pipe->eof()) {
            //~ struct timeval t1;
            //~ ::gettimeofday(&t1, 0);
            //~ // streaming mode - we do not flush here
            //~ *pipe >> s;
            //~ if (!*pipe || pipe->eof()) break;
            //~ struct timeval t2;
            //~ ::gettimeofday(&t2, 0);
            //~ t2.tv_sec -= t1.tv_sec;
            //~ t2.tv_usec -= t1.tv_usec;
            //~ double dt = 1e-6 * double(t2.tv_usec) + double(t2.tv_sec);
            //~ if (std::strlen(s)) {
                //~ ++n;
                //~ if (dt < min) min = dt;
                //~ if (dt > max) max = dt;
                //~ avg += dt;
                //~ bsz = std::strlen(s);
            //~ } else {
                //~ if (!n) break;
                //~ // next block size
                //~ avg /= double(n);
                //~ avg *= 1e6; min *= 1e6; max *= 1e6;
//~ 
                //~ std::cout << "block size " << std::setw(9) << bsz <<
                    //~ " avg " << std::setw(7) << avg << " us min " <<
                    //~ std::setw(7) << min << " us max " << std::setw(7) <<
                    //~ max << "us speed " << std::setw(9) <<
                    //~ (double(bsz) / double(1 << 20) / (1e-6 * avg)) <<
                    //~ " MB/s" << std::endl;
                //~ n = 0;
                //~ avg = 0.;
                //~ min = 1e42;
                //~ max = -1e42;
            //~ }
        //~ }
        //~ int retVal = pipe->close();
            //~ std::cout << "[PARENT]: child exited with code " << retVal << std::endl;
        //~ if (retVal) return retVal;
        //~ delete pipe;
        //~ std::free(s);
    //~ }
    //~ return 0;

// additional tests

int simplechild_(BidirMMapPipe& pipe)
{
    // child does an echo loop
    while (pipe.good() && !pipe.eof()) {
        // read a string
        std::string str;
        pipe >> str;
        if (!pipe) return -1;
        if (pipe.eof()) break;
        if (!str.empty()) {
            std::cout << "[CHILD] :  read: " << str << "\n";
            str = "... early in the morning?";
        }
        pipe << str << BidirMMapPipe::flush;
        // did our parent tell us to shut down?
        if (str.empty()) break;
        if (!pipe) return -1;
        if (pipe.eof()) break;
        std::cout << "[CHILD] : wrote: " << str << "\n";
    }
    std::cout << "[CHILD] : shutting down "  << pipe.isParent() << std::endl;
    std::cout << "[CHILD] : close " << pipe.close() << std::endl;
    return 0;
}

BidirMMapPipe* spawnChild_(int (*childexec)(BidirMMapPipe&))
{
    // create a pipe with the given child at the remote end
    BidirMMapPipe *p = new BidirMMapPipe();
    if (p->isChild()) {
        int retVal = childexec(*p);
        delete p;
        std::exit(retVal);
    }
    return p;
}

int simplerelay(BidirMMapPipe& in,BidirMMapPipe& out)
{
    // child does an echo loop
    while (in.good() && !in.eof()) {
        // read a string
        std::string str;
        in >> str;
        if (!in) return -1;
        if (in.eof()) break;
        //~ if (!str.empty()) {
            std::cout << "[RELAY] :  from in: " << str << "\n";
            out << str << BidirMMapPipe::flush;
            std::cout << "[RELAY] : to out: " << str << "\n";
        //~ }
        // did our parent tell us to shut down?
        if (str.empty()) break;
        if (!out) return -1;
        if (out.eof()) break;
        out >> str;
        if (!in) return -1;
        if (in.eof()) break;
        //~ if (!str.empty()) {
            std::cout << "[RELAY] :  from out: " << str << "\n";
            in << str << BidirMMapPipe::flush;
            std::cout << "[RELAY] :  to in: " << str << "\n";
        //~ }
    }
    std::cout << "[RELAY] : shutting down "  << std::endl;
    return 0;
}


#include <sys/time.h>
#include <iomanip>

// simple echo loop test
template<class T>
int simple_echo(T& out, BidirMMapPipe* pipe)
{
    {
        out << "[PARENT]: simple challenge-response test, "
            "one child:" << "\n";
        for (int i = 0; i < 2; ++i) {
            std::string str("What shall we do with a drunken sailor...");
            *pipe << str + "  "+ std::to_string(i) << BidirMMapPipe::flush;
            if (!*pipe) return -1;
            out << "[PARENT]: wrote: " << str << "\n";
            *pipe >> str;
            if (!*pipe) return -1;
            out << "[PARENT]:  read: " << str << "\n";
            out.flush();            
        }
        // send shutdown string
        *pipe << "" << BidirMMapPipe::flush;
        // wait for shutdown handshake
        std::string s;
        *pipe >> s;
        int retVal = pipe->close();
        out << "[PARENT]: status of child: " << std::to_string(retVal) <<
            "\n";
        out.flush();
        return retVal;
    }  
}

void simple_echo_direct()
{
    BidirMMapPipe* pipe = spawnChild_(simplechild_);
    simple_echo(std::cout, pipe);
    delete pipe;
}

void simple_echo_relay()
{
    BidirMMapPipe* pipe1 = spawnChild(simplechild_);
    BidirMMapPipe* pipe2 = new BidirMMapPipe(true, false, false);

    if(pipe2->isChild()) {
      std :: cout << "child of p2 considers itself pipe parent? " << pipe1->isParent() << std:: endl;
      simplerelay(*pipe2, *pipe1); // forward output over pipe2
      pipe2->close();
      pipe1->close();
    }
    if(pipe2->isParent()) {
      simple_echo(std::cout, pipe2);
    }
    std::cout << "ending" << std::endl;
}

TEST(testBidirMMapPipe_direct, getResult)
{
    simple_echo_direct();
}


TEST(testBidirMMapPipe_relay, getResult)
{
    simple_echo_relay();
}


TEST(testBidirMMapPipe_both, getResult)
{
    simple_echo_direct();
    simple_echo_relay();
    simple_echo_direct();

}
