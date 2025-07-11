import RPi.GPIO as GPIO
import socket

from utils.LARS_utils import kwVectorHelper
from ctypes import *

class BinaryAction:
    def __init__(self, on_voltage, pins):
        """
            on_voltage - the setting to set the pin to turn this action on (HIGH or LOW)
            pins - an iterable of the pins controlled by this action 
        """

        self.pins = pins
        self.state = False # intialize to off
        self.on_voltage = on_voltage

        if self.on_voltage == HIGH:
            self.off_voltage = LOW
        else:
            self.off_voltage = HIGH
        
    def on(self):
        self.state = True

        for pin in self.pins:
            GPIO.output(pin, self.on_voltage)
    
    def off(self):
        self.state = False

        for pin in self.pins
            GPIO.output(pin, self.off_voltage)

    def flip_state(self):
        for pin in self.pins:
            GPIO.output(pin, self.on_voltage if self.state else self.off_voltage)
        
        self.state = not self.state

# intialize c++ functions
#clib_serial = CDLL("utils/serialModule.so")

#clib_serial.openSerialPort.argtypes = [c_char_p]
#clib_serial.openSerialPort.restype = c_int
#clib_serial.configureSerialPort.argtypes = [c_int, c_int, c_int]
#clib_serial.configureSerialPort.restype = c_bool
#clib_serial.readSerial.argtypes = [c_int, c_size_t, POINTER(c_ssize_t)]
#clib_serial.readSerial.restype = POINTER(c_bool)

GPIO.setmode(GPIO.BOARD) # use physical board numbering

#serial_portname = ""
#serial_speed = 115200
#expected_serial_bytes = 2
#total_read = POINTER(c_ssize_t(0))

overhead_lamp_pin = 15
desk_lights_pin = 18

GPIO.setup(overhead_lamp_pin, GPIO.OUT)
GPIO.setup(desk_lights_pin, GPIO.OUT)

overhead_lamp = BinaryAction((overhead_lamp_pin,), LOW)
desk_lights = BinaryAction((desk_lights_pin,), LOW)
all_lights = BinaryAction([desk_lights_pin, overhead_lamp_pin], LOW)

kw_helper = kwVectorHelper()
kw_encodings = kw_helper.get_encodings()
# switch the values for keys and keys for values so we can get an action by its index
kw_encodings = {
    value: key for key, value in kw_encodings.items()
    }

kw_to_action = {
    "lights": all_lights.flip_state,
    "lights on": all_lights.on,
    "lights off": all_lights.off,
    "overhead lamp off": overhead_lamp.off,
    "overhead lamp on": overhead_lamp.on,
    "desk lights off": desk_lights.off,
    "desk lights on": desk_lights.on 
}

running = True

if __name__ == "__main__":
    #serial = clib_serial.openSerialPort(serial_portname)
    #if serial == -1:
    #    raise("There was an issue starting the serial port: {}".format(serial_portname))
    #if not clib_serial.configureSerialPort(serial, serial_speed, expected_serial_bytes):
    #    raise("There was an issue configuring the serial port")

    try:
        while running:
            #rec_packet = clib_serial.readSerial(serial, expected_serial_bytes, total_read)
            #if not rec_packet:
            #    print("There was an issue reading serial port")
            #    continue

            packet = packet.split(",")
            for i in packet:
                keyword = kw_encodings[i]
                kw_to_action[keyword]() # execute action

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False