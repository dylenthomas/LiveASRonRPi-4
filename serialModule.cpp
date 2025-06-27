#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <unistd.h>
using namespace std;

// g++ -fPIC -shared -o utils/serialModule.so serialModule.cpp -lboost_system -lpthread

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
    int readSerial(int fd, char* buffer, size_t n) {
        ssize_t total_read = 0;
        while (total_read < n) {
            ssize_t bytes = read(fd, buffer + total_read, n - total_read);
            // buffer + total_read moves the pointer forward in buffer by total_read bytes
            if (bytes < 0) {
                cerr << "Error when reading serial buffer: " << strerror(errno) << endl;
                return -1;
            }
        total_read += bytes;
        }
        return static_cast<int>(total_read);
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