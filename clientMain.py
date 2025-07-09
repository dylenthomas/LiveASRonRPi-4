import RPi.GPIO as GPIO

from utils.kwHelper import kwVectorHelper
from ctypes import *

class kwActionTranslator:
    """
    Convert the recieved boolean array representing the bitfield recieved and translate that into actions
    """
    def __init__(self)
        #pin, on or off (false = off, true = on, none = binary switch)
        kw_helper = kwVectorHelper()
        self.cmd_encodings = kw_helper.get_encodings()
        # switch the values for keys and keys for values so we can get a action by its index
        self.cmd_encodings = {
            value: key for key, value in self.cmd_encodings.items()
        }
        
        # encode commands to a bool or none value if the command is inheriently binary
        # none means the command is binary so it just switches the current state
        # false means the command turns a pin low
        # true means the command turn a pin high
        self.binary_actions {
            "lights": None,
            "lights on": True,
            "lights off": False,
            "overhead lamp off": False,
            "overhead lamp on": True,
            "desk lights off": False,
            "desk lights on": True,
        }

        # encode pin actions to their pin numbers
        self.pins = {
            "lights": 16,
            "lights on": 16,
            "lights off": 16,
            "overhead lamp off": 15,
            "overhead lamp on": 15
            "desk lights off": 18,
            "desk lights on": 18
        }

        # keep track of the state of the pins
        self.pin_states = {
            "16": False,
            "15": False,
            "18": False
        }

    def parseKWS(self, packet):
        actions = [] # pins and a corresponding action
        packet = packet.split(',') # get as a list of strings by splitting at the comma values

        # use the int from packet which corresponds to a command
        for i in packet:
            # get the command name and its pin
            cmd = self.cmd_encodings[i]
            pin = self.pins[cmd]

            # get the type of action
            act = 'not binary'
            for k in self.binary_actions.keys():
                if cmd == k:
                    act = self.binary_actions[k]
            
            # check act was assigned a binary action
            if act == 'not binary':
                continue

            # if action is binary flip current state
            state = self.pin_states[str(pin)]
            if act == None:
                # swap the current state
                new_state = not state
                actions.append((pin, new_state))
            else:
                # just send the bool value of act
                new_state = act
                actions.append((pin, new_state))

            self.pin_states[str(pin)] = new_state # update pin states

        return actions 

# intialize c++ functions
clib_serial = CDLL("utils/serialModule.so")

clib_serial.openSerialPort.argtypes = [c_char_p]
clib_serial.openSerialPort.restype = c_int
clib_serial.configureSerialPort.argtypes = [c_int, c_int, c_int]
clib_serial.configureSerialPort.restype = c_bool
clib_serial.readSerial.argtypes = [c_int, c_size_t, POINTER(c_ssize_t)]
clib_serial.readSerial.restype = POINTER(c_bool)

# initialize pins
kw_translator = kwActionTranslator()
pins = kw_translator.pins

GPIO.setmode(GPIO.BOARD) # use physical board numbering
GPIO.setup(pins["lights"], GPIO.OUT)
GPIO.setup(pins["overhead lamp"], GPIO.OUT)
GPIO.setup(pins["desk lights"], GPIO.OUT)

serial_portname = ""
serial_speed = 115200
expected_serial_bytes = 2
total_read = POINTER(c_ssize_t(0))

running = True

if __name__ == "__main__":
    serial = clib_serial.openSerialPort(serial_portname)
    if serial == -1:
        raise("There was an issue starting the serial port: {}".format(serial_portname))
    if not clib_serial.configureSerialPort(serial, serial_speed, expected_serial_bytes):
        raise("There was an issue configuring the serial port")

    try:
        while running:
            rec_packet = clib_serial.readSerial(serial, expected_serial_bytes, total_read)
            if not rec_packet:
                print("There was an issue reading serial port")
                continue

            actions = kw_translator.parseKWS(rec_packet)
            for action in actions:
                p = actions[0] # pin for the current action
                tg = actions[1] # bool indicating how to change the state
                if tg: GPIO.output(p, GPIO.HIGH)
                else: GPIO.output(p, GPIO.LOW)

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False