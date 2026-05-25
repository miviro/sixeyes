PORT   ?= /dev/ttyUSB0
FQBN   ?= esp32:esp32:esp32
SKETCH := src/esp32
SCRIPT := src/host/sixeyes.py

.PHONY: run flash start sync

all: flash start

flash:
	arduino-cli compile --fqbn $(FQBN) $(SKETCH)
	arduino-cli upload  --fqbn $(FQBN) --port $(PORT) $(SKETCH)

start:
	uv run python $(SCRIPT)
