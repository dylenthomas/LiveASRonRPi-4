import RPi.GPIO as GPIO

from utils.LARS_utils import kwVectorHelper, TCPCommunication

### AMP Remote scancodes for ir-ctl
# Kenwood RC-80 IR codes (protocol: NEC)
# To send IR Blast for a specific code:
#   sudo ir-ctl -S nec:scancode
#
#   They can be chained together:
#   sudo ir-ctl -S nec:scancode1 -S nec:scancode2
#
#   You can also add repeats to simulate holding a button
#   sudo ir-ctl -S nec:scancode:n

CODES = {
    "Audio 1": 0xb881,
    "Audio 2": 0xb882,
    "Audio 3": 0xb883,
    "Audio 4": 0xb884,
    "Audio 5": 0xb885,
    "Audio 6": 0xb886,
    "Audio 7": 0xb887,
    "Audio 8": 0xb888,
    "Audio 9": 0xb889,
    "Audio 0": 0xb880,

    "TV/Video 1": 0x411,
    "TV/Video 2": 0x412,
    "TV/Video 3": 0x413,
    "TV/Video 4": 0x414,
    "TV/Video 5": 0x415,
    "TV/Video 6": 0x416,
    "TV/Video 7": 0x417,
    "TV/Video 8": 0x418,
    "TV/Video 9": 0x419,
    "TV/Video 0": 0x410,

    "+10": 0xb80d,

    "CD Pause/Play": 0xb8cb,
    "CD Stop": 0xb8c9,
    "CD Disc": 0xb808,
    "CD Skip Back": 0xb8ce,
    "CD Skip Forward": 0xb8cf,
    "CD Search Back": 0xb806,
    "CD Search Forward": 0xb807,

    "PHONO Back": 0xb8c1,
    "PHONO Stop": 0xb8c0,

    "TUNER Direct": 0xb89e,
    "TUNER A/B": 0xb89f,
    "TUNER P Scan": 0xb899,
    "TUNER FM": 0xb88f,
    "TUNER AM": 0xb88e,

    "TAPE A Rec": 0xb8d6,
    "TAPE A/TAPE B/VIDEO Skip Back": 0xb8da,
    "TAPE A/TAPE B/VIDEO Back": 0xb8d8,
    "TAPE A/TAPE B/VIDEO Forward": 0xb8d9,
    "TAPE A/TAPE B/VIDEO Skip Forward": 0xb8db,
    "TAPE B/VIDEO Rec": 0xb8de,
    "TAPE A/TAPE B/VIDEO Stop": 0xb8dd,
    "TAPE A/TAPE B/VIDEO Pause": 0xb8dc,

    "CD": 0xb892,
    "PHONO": 0xb890,
    "TUNER": 0xb891,
    "TAPE 1": 0xb894,
    "TAPE 2": 0xb895,
    "VIDEO 1": 0xb896,
    "VIDEO 2": 0xb893,
    "VIDEO 3": 0xb88a,

    "SYSTEM MEMORY 1": 0xb840,
    "SYSTEM MEMORY 2": 0xb841,

    "EQ.": 0xb8c5,
    "SURROUND": 0xb8d7,
    "MUTE": 0xb89c,
    "AUDIO POWER": 0xb89d,

    "TV POWER": 0x408,
    "VIDEO POWER": 0x1908,

    "TV CH. Up": 0x400,
    "TV CH. Down": 0x401,
    "TV": 0x409,
    "VIDEO": 0x190a,
    "TV VOL. Up": 0x402,
    "TV VOL. Down": 0x403,

    "REAR LEVEL Up": 0xb8c7,
    "REAR LEVEL Down": 0xb8c6,
    "MAIN VOL. Up": 0xb89b,
    "MAIN VOL. Down": 0xb89a,
}
###

class BinaryAction:
    def __init__(self, pins, on_voltage):
        """
            on_voltage - the setting to set the pin to turn this action on (HIGH or LOW)
            pins - an iterable of the pins controlled by this action 
        """

        self.pins = pins
        self.state = False # intialize to off
        self.on_voltage = on_voltage

        if self.on_voltage == GPIO.HIGH:
            self.off_voltage = GPIO.LOW
        else:
            self.off_voltage = GPIO.HIGH
        
    def on(self):
        self.state = True

        for pin in self.pins:
            GPIO.output(pin, self.on_voltage)
    
    def off(self):
        self.state = False

        for pin in self.pins:
            GPIO.output(pin, self.off_voltage)

    def flip_state(self):
        for pin in self.pins:
            GPIO.output(pin, self.on_voltage if self.state else self.off_voltage)
        
        self.state = not self.state


### Setup GPIO Pins ---
GPIO.setmode(GPIO.BOARD) # use physical board numbering

overhead_lamp_pin = 15
desk_lights_pin = 18

GPIO.setup(overhead_lamp_pin, GPIO.OUT)
GPIO.setup(desk_lights_pin, GPIO.OUT)
###

### Setup Actions ---
overhead_lamp = BinaryAction((overhead_lamp_pin,), GPIO.LOW)
desk_lights = BinaryAction((desk_lights_pin,), GPIO.LOW)
all_lights = BinaryAction([desk_lights_pin, overhead_lamp_pin], GPIO.LOW)
###

kw_helper = kwVectorHelper()
kw_encodings = kw_helper.get_encodings()
# switch the values for keys and keys for values so we can get an action by its index
kw_encodings = {
    str(value): key for key, value in kw_encodings.items()
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

tcpCommunicator = TCPCommunication()

if __name__ == "__main__":
    print("Starting tcp server...")
    tcpCommunicator.openServer()

    try:
        while running:
            packet = tcpCommunicator.readFromClient()
            print("Recieved packet: ", packet)

            if packet is None: continue
            
            packet = packet.split(",")
            for i in packet:
                if i == '': continue # skip the empty entry at the end
                keyword = kw_encodings[i]
                kw_to_action[keyword]() # execute action

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        tcpCommunicator.closeClientConnection()
        GPIO.cleanup()
