import RPi.GPIO as GPIO

from ctypes import *

class kwActionTranslator:
    """
    Convert the recieved boolean array representing the bitfield recieved and translate that into actions
    """
    def __init__(self, pins_encoding)
        #pin, on or off (false = off, true = on, none = binary switch)
        self.encoding = {
            "0": (15, False),
            "1": (15, True),
            "2": (18, False),
            "3": (18, True),
            "4": (),
            "5": (),
            "6": (16, True),
            "7": (16, False),
            "8": (),
            "9": (),
            "10": (16, None),
            "11": (),
            "12": ()
        }

        # encode pin actions to their pin numbers
        self.pins = {
            "lights": 16,
            "overhead lamp": 15,
            "desk lights": 18
        }

        # keep track of the state of the pins
        self.pin_states = {
            "16": None,
            "15": None,
            "18": None
        }

    def parseKWS(self, bool_arr):
        actions = [] # pins/things that need to be actviated

        for i, bl in enumerate(bool_arr):
            if bl:
                act = self.encoding[str(i)]
                # if a binary action check the current state
                if act[1] is None:
                    state = self.pin_states[str(act[0])]
                    if state is None or not state:
                        act[1] = True
                    else:
                        act[1] = False

                actions.append(act)
                self.pin_states[str(act[0])] = act[1] # update pin states

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
            cmd_bool = clib_serial.readSerial(serial, expected_serial_bytes, total_read)
            if not cmd_bool:
                print("There was an issue reading serial port")
                continue

            actions = kw_translator.parseKWS(cmd_bool)
            for action in actions:
                p = actions[0]
                tg = actions[1]
                if tg: GPIO.output(p, GPIO.HIGH)
                else: GPIO.output(p, GPIO.LOW)

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False