from __future__ import annotations

import time

import serial

from .config import clamp


class ESP32SerialLink:
    def __init__(self, port: str, baud_rate: int):
        self.serial = serial.Serial(
            port,
            baud_rate,
            timeout=0.05,
            write_timeout=0.05,
            inter_byte_timeout=0.0,
        )
        time.sleep(2.0)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def send(self, pitch: float, yaw: float) -> None:
        pitch_byte = int(round(clamp(pitch, 0.0, 180.0)))
        yaw_byte = int(round(clamp(yaw, 0.0, 180.0)))
        self.serial.reset_output_buffer()
        self.serial.write(bytes([0xFF, pitch_byte, yaw_byte]))
        self.serial.flush()

    def close(self) -> None:
        if not self.serial.is_open:
            return
        try:
            self.serial.reset_output_buffer()
            self.serial.reset_input_buffer()
        except serial.SerialException:
            pass
        self.serial.close()
