#include <Arduino.h>

const int pitchPin = 2;
const int yawPin = 4;
const int freq = 50;
const int resolution = 12;

float currentPitch = 90.0;
float currentYaw = 90.0;
int targetPitch = 90;
int targetYaw = 90;
const float stepSize = 0.5;

unsigned long lastBroadcast = 0;
const unsigned long broadcastInterval = 50; // Broadcast every 50ms (20Hz)

void updateServo(int pin, float angle) {
  angle = constrain(angle, 0, 180);
  long pulseUs = map(angle, 0, 180, 500, 2400);
  uint32_t dutySteps = (pulseUs * 4095) / 20000;
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
  // 1. Receive Target
  if (Serial.available() > 0) {
    targetPitch = Serial.parseInt();
    if (Serial.read() == ',') {
      targetYaw = Serial.parseInt();
    }
    while(Serial.available() > 0) Serial.read();
  }

  // 2. Smooth Move Logic
  if (abs(currentPitch - targetPitch) > stepSize) {
    currentPitch += (targetPitch > currentPitch) ? stepSize : -stepSize;
    updateServo(pitchPin, currentPitch);
  }
  if (abs(currentYaw - targetYaw) > stepSize) {
    currentYaw += (targetYaw > currentYaw) ? stepSize : -stepSize;
    updateServo(yawPin, currentYaw);
  }

  // 3. Continuous Broadcast (Non-blocking)
  if (millis() - lastBroadcast >= broadcastInterval) {
    // Format: P:angle,Y:angle
    Serial.print("P:");
    Serial.print(currentPitch, 1);
    Serial.print(",Y:");
    Serial.println(currentYaw, 1);
    lastBroadcast = millis();
  }

  delay(10); 
}