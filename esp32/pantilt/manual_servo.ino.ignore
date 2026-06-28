#include <Arduino.h>

#define PIN        4    // yawPin=2  pitchPin=4
#define STEP       10   // degrees per Enter press
#define START_DEG  0

const int freq       = 50;
const int resolution = 12;

int currentAngle = START_DEG;

void updateServo(int pin, float angle) {
  angle = constrain(angle, 0.0f, 180.0f);
  long pulseUs  = (long)(500.0f + (angle / 180.0f) * 1900.0f);
  uint32_t duty = (pulseUs * 4095UL) / 20000UL;
  ledcWrite(pin, 4095 - duty);
}

void setup() {
  Serial.begin(115200);
  ledcAttach(PIN, freq, resolution);
  updateServo(PIN, currentAngle);
  Serial.print("Start: "); Serial.println(currentAngle);
  Serial.println("Press Enter to step +10 deg");
}

void loop() {
  if (Serial.available()) {
    while (Serial.available()) Serial.read(); // flush
    currentAngle += STEP;
    if (currentAngle > 180) currentAngle = 0; // wrap
    updateServo(PIN, currentAngle);
    Serial.print("Angle: "); Serial.println(currentAngle);
  }
}
