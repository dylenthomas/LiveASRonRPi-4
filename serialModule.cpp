#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>
using namespace std;

// g++-9 -fPIC -shared -o utils/serialModule.so serialModule.cpp

extern "C" {
    int openSerialPort(const char* portname) {
        // https://www.man7.org/linux/man-pages/man2/open.2.html
        int fd = open(portname, O_RDWR | O_NOCTTY | O_SYNC);
        // O_NOCTTY - program does not want to control the terminal on this port
        // O_SYNC - all rwite operations to the serial will be completed according to requirements of synchronized I/O file integrity
        // O_RDWR - set file for read and write

        if (fd < 0) {
            cerr << "Error opening " << portname << ": " << strerror(errno) << endl;
            return -1;
        }

        return fd;
    }
}

extern "C" {
    bool configureSerialPort(int fd, int speed, int n) {
        struct termios tty;
        if (tcgetattr(fd, &tty) != 0) {
            cerr << "Error from tcgetattr: " << strerror(errno) << endl;
            return false;
        }

        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // set the control mode to 8 bits
        tty.c_iflag &= ~IGNBRK; // ignore break condition on the input
        tty.c_lflag = 0; // don't enable any of the local modes
        tty.c_oflag = 0; // don't enable any of the output modes
        // configure for blocking read
        // read() will wait for n bytes to be available in the buffer before reading
        tty.c_cc[VMIN] = n;
        tty.c_cc[VTIME] = 0;
        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // disable control flow so data can be sent and recieved without being paused or resumed
        tty.c_cflag |= (CLOCAL | CREAD); // ignore modem control lines and enable receiever
        tty.c_cflag &= ~(PARENB | PARODD); // disable parity generation and checking 
        tty.c_cflag &= ~CSTOPB; // use one stop bit
        tty.c_cflag &= ~CRTSCTS; // don't use RTS/CTS flow control

        if (tcsetattr(fd, TCSANOW, &tty) != 0) {
            cerr << "Error from tcsetattr: " << strerror(errno) << endl;
            return false;
        }

        return true;
    }
}

extern "C" {
    bool* readSerial(int fd, size_t n, ssize_t* total_read) {
        // function to read the serial port for this specific application
        int read_bytes = 0; // internally keep track of read bytes
        char buffer[n];

        while (read_bytes < n) {
            ssize_t bytes = read(fd, buffer + read_bytes, n - read_bytes);
            // buffer + read_bytes moves the pointer forward in buffer by read_bytes bytes
            if (bytes < 0) {
                cerr << "Error when reading serial buffer: " << strerror(errno) << endl;
                return nullptr; // in python must check "if not result" where result would be the nullptr if if failed to read
            }
        read_bytes += bytes;
        }
        *total_read = read_bytes; // tell the python program how many bytes where read

        // since bytes is a bitfiled, mask it to get the intended commands (Big Endian format)
        size_t total_bits = read_bytes * 8;
        bool* bit_array = new bool[total_bits];
        
        for (size_t byte_ind = 0; byte_ind < read_bytes; byte_ind++) {
            unsigned char byte = static_cast<unsigned char>(buffer[byte_ind]);
            for (size_t b = 0; b < 8; b++) {
                // mask Big Endian formated bytes
                bit_array[b + 8 * byte_ind] = (byte & (0x01 << (7 - b))) == 1;
            }
            
        }
        
        return bit_array;
    }
}

extern "C" {
    int writeSerial(int fd, const char* buffer, size_t n) {
        ssize_t sent_bytes = write(fd, buffer, n);
        if (sent_bytes == -1) {
            cerr << "Error sending data to serial" << strerror(errno) << endl;
            return -1;
        }
        return static_cast<int>(sent_bytes);
    }
}

extern "C" {
    void closeSerial(int fd) {close(fd);}
}