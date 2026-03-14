PORT   ?= /dev/ttyUSB0
FQBN   ?= esp32:esp32:esp32
SKETCH := esp32
SCRIPT := src/sixeyes.py
PYTHON ?= python3

.PHONY: run flash start

all: flash start

flash:
	arduino-cli compile --fqbn $(FQBN) $(SKETCH)
	arduino-cli upload  --fqbn $(FQBN) --port $(PORT) $(SKETCH)

start:
	$(PYTHON) $(SCRIPT)
