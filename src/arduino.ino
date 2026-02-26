#include <Arduino.h>

const int pitchPin = 4;
const int yawPin = 2;
const int freq = 50;
const int resolution = 12;

float currentPitch = 90.0;
float currentYaw = 90.0;
float targetPitch = 90.0;
float targetYaw = 90.0;

// Deadband: within this range, don't move (kills micro-jitter)
const float DEADBAND = 5.0f;

// Physical travel limits
const float PITCH_MIN = 20.0f;
const float PITCH_MAX = 160.0f;
const float YAW_MIN   = 20.0f;
const float YAW_MAX   = 160.0f;

unsigned long lastBroadcast = 0;
const unsigned long broadcastInterval = 50; // Broadcast every 50ms (20Hz)

void updateServo(int pin, float angle) {
  angle = constrain(angle, 0.0f, 180.0f);
  // Float-safe pulse calculation (avoids integer truncation from map())
  long pulseUs = (long)(500.0f + (angle / 180.0f) * 1900.0f);
  uint32_t dutySteps = (pulseUs * 4095UL) / 20000UL;
  ledcWrite(pin, 4095 - dutySteps);
}


void setup() {
  Serial.begin(115200);
  ledcAttach(pitchPin, freq, resolution);
  ledcAttach(yawPin, freq, resolution);
  updateServo(pitchPin, currentPitch);
  updateServo(yawPin, currentYaw);
}

void loop() {
  // 1. Receive Target (format: "pitch,yaw\n")
  if (Serial.available() > 0) {
    targetPitch = constrain((float)Serial.parseInt(), PITCH_MIN, PITCH_MAX);
    if (Serial.read() == ',') {
      targetYaw = constrain((float)Serial.parseInt(), YAW_MIN, YAW_MAX);
    }
    while (Serial.available() > 0) Serial.read();
  }

  // 2. Instant move with deadband
  if (fabsf(targetPitch - currentPitch) > DEADBAND) {
    currentPitch = targetPitch;
    updateServo(pitchPin, currentPitch);
  }
  if (fabsf(targetYaw - currentYaw) > DEADBAND) {
    currentYaw = targetYaw;
    updateServo(yawPin, currentYaw);
  }

  // 3. Continuous Broadcast (format: "P:angle,Y:angle")
  if (millis() - lastBroadcast >= broadcastInterval) {
    Serial.print("P:");
    Serial.print(currentPitch, 1);
    Serial.print(",Y:");
    Serial.println(currentYaw, 1);
    lastBroadcast = millis();
  }

  delay(10);
}
