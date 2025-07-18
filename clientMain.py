import RPi.GPIO as GPIO

from utils.LARS_utils import kwVectorHelper, TCPCommunication

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
            print("Packet: ", packet)

            if packet is None: continue
            
            packet = packet.split(",")
            for i in packet:
                keyword = kw_encodings[i]
                kw_to_action[keyword]() # execute action

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        tcpCommunicator.closeClientConnection()