#include <Arduino.h>

const int pitchPin   = 2;
const int yawPin     = 4;
const int freq       = 50;
const int resolution = 12;

const float PITCH_MIN = 0.0f;
const float PITCH_MAX = 140.0f;
const float YAW_MIN   = 0.0f;
const float YAW_MAX   = 160.0f;

// Precomputed servo constants  (duty = DUTY_OFFSET + angle * DUTY_SCALE)
// pulseUs  = 500 + angle * (1900/180)
// duty     = pulseUs * 4095 / 20000
// => duty  = 102.375 + angle * 2.16042
static const float DUTY_OFFSET = 102.375f;
static const float DUTY_SCALE  = 2.16042f;  // (1900.0/180.0) * (4095.0/20000.0)

inline void writeServo(int pin, float angle) {
  uint32_t duty = (uint32_t)(DUTY_OFFSET + angle * DUTY_SCALE);
  ledcWrite(pin, 4095 - duty);
}

// ---------------------------------------------------------------------------
// Non-blocking binary parser
// Protocol: 3 bytes — 0xFF (sync) | pitch (0-180) | yaw (0-180)
// 0xFF is the sync marker: safely outside the valid angle range (0-180)
// ---------------------------------------------------------------------------
// Stop PWM after this many ms of silence → servo goes quiet and unpowered
const unsigned long IDLE_TIMEOUT_MS = 500;

enum ParseState : uint8_t { WAIT_SYNC, READ_PITCH, READ_YAW };
static ParseState state       = WAIT_SYNC;
static float      parsedPitch = 90.0f;
static bool       active      = false;
static unsigned long lastFrameMs = 0;

inline void detachServos() {
  ledcWrite(pitchPin, 0);
  ledcWrite(yawPin,   0);
  active = false;
}

void setup() {
  Serial.begin(115200);
  ledcAttach(pitchPin, freq, resolution);
  ledcAttach(yawPin,   freq, resolution);
  writeServo(pitchPin, 90.0f);
  writeServo(yawPin,   90.0f);
  lastFrameMs = millis();
}

void loop() {
  while (Serial.available()) {
    uint8_t b = Serial.read();

    switch (state) {
      case WAIT_SYNC:
        if (b == 0xFF) state = READ_PITCH;
        break;

      case READ_PITCH:
        parsedPitch = constrain((float)b, PITCH_MIN, PITCH_MAX);
        state = READ_YAW;
        break;

      case READ_YAW:
        writeServo(pitchPin, parsedPitch);
        writeServo(yawPin, constrain((float)b, YAW_MIN, YAW_MAX));
        lastFrameMs = millis();
        active      = true;
        state       = WAIT_SYNC;
        break;
    }
  }

  if (active && (millis() - lastFrameMs > IDLE_TIMEOUT_MS))
    detachServos();
}
