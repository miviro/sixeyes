PORT      ?= /dev/ttyUSB0
FQBN      ?= esp32:esp32:esp32
CAM_FQBN  ?= esp32:esp32:esp32cam
CAM_PORT  ?= /dev/ttyUSB0
SKETCH    := esp32/pantilt
CAM_SKETCH := esp32/eighteyes
SCRIPT    := src/sixeyes.py
MODEL     ?= yolo11m.pt

.PHONY: run flash flash-cam start sync

all: flash start

flash:
	arduino-cli compile --fqbn $(FQBN) $(SKETCH)
	arduino-cli upload  --fqbn $(FQBN) --port $(PORT) $(SKETCH)

flash-cam:
	arduino-cli compile --upload \
		--fqbn $(CAM_FQBN) \
		--build-property "build.partitions=" \
		--build-property "upload.maximum_size=16777216" \
		--port $(CAM_PORT) \
		$(CAM_SKETCH)

start:
	uv run python $(SCRIPT) $(MODEL) eighteyes
