/**
 * KRISHI-EYE Teensy 4.1 Firmware — Precision Spray Controller
 * * UART Interface: Pins 0 (RX1) and 1 (TX1) @ 115200 Baud
 * Protocol: 10 fields + optional OLED text (space separated, ends with \n)
 * Format: <pwm1> <pwm2> <sc1> <sc2> <steps> <v1> <v2> <v3> <v4> <buzzer> [oled]
 */

#include <Wire.h>
#include <PWMServo.h>
#include <SparkFun_u-blox_GNSS_v3.h>
#include "teensystep4.h"
#include <SCServo.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

using namespace TS4;

// ─────────────────────────────────────────────────────────────
// PIN DEFINITIONS
// ─────────────────────────────────────────────────────────────
#define RPI_SERIAL      Serial1
#define RPI_BAUD        115200

#define SERVO1_PIN      24
#define SERVO2_PIN      25

#define STEP_PIN        4
#define DIR_PIN         5
#define EN_PIN          6

const int VALVE_PIN[4] = {26, 27, 28, 29};
#define NUM_VALVES 4

#define SC_BAUD_RATE    1000000
#define SC_NUM_SERVOS   2

#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT    64
#define OLED_ADDR       0x3C

#define BUZZER_PIN      40

// ─────────────────────────────────────────────────────────────
// CONSTRAINTS & CALIBRATION
// ─────────────────────────────────────────────────────────────
#define PWM1_MIN   30    // Camera H Safe Min
#define PWM1_MAX  150    // Camera H Safe Max
#define PWM2_MIN   10    // Camera V Safe Min
#define PWM2_MAX   80    // Camera V Safe Max
#define S1_HOME    90
#define S2_HOME    45

#define SC_MIN     30    // Nozzle H/V Safe Min
#define SC_MAX    100    // Nozzle H/V Safe Max (Joints break > 100)

#define TANK_CAPACITY_ML  100.0
#define LOW_TANK_ML       20.0
#define PULSES_PER_ML     160.0

// ─────────────────────────────────────────────────────────────
// STATE VARIABLES
// ─────────────────────────────────────────────────────────────
SFE_UBLOX_GNSS    myGNSS;
PWMServo          servo1, servo2;
Stepper           stepper(STEP_PIN, DIR_PIN);
SMS_STS           scServo;
Adafruit_SH1106G  oled(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire1, -1);

bool oledOK = false;
int  lastS1 = S1_HOME, lastS2 = S2_HOME;
int  lastScAngle[2] = {65, 65}; // Middle of 30-100 safe range
long totalSteps = 0;
bool valveState[4] = {false, false, false, false};
bool bzState = false;

float tankLevel[4] = {TANK_CAPACITY_ML, TANK_CAPACITY_ML, TANK_CAPACITY_ML, TANK_CAPACITY_ML};
const char* TANK_NAMES[4] = {"BACT", "FUNG", "NEMA", "PEST"};

bool dispensing = false;
int  activeValveIdx = -1;
float currentDispenseML = 0;

struct { bool valid = false; double lat, lon; int sats; } gnss;
elapsedMillis gnssTimer, scFeedbackTimer, stepperIdleTimer;

// ─────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────
void dualPrintf(const char* fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Serial.print(buf);
  RPI_SERIAL.print(buf);
}

void oledStatus(String msg) {
  if (!oledOK) return;
  oled.clearDisplay();
  oled.fillRect(0, 0, 128, 13, SH110X_WHITE);
  oled.setTextColor(SH110X_BLACK);
  oled.setCursor(4, 3); oled.print("KRISHI-EYE CTRL");
  oled.setTextColor(SH110X_WHITE);
  oled.setCursor(0, 25); oled.setTextSize(2); oled.print(msg);
  oled.display();
}

void calibratePWM() {
  Serial.println(">> STARTING STARTUP SWEEP...");
  servo1.write(PWM1_MIN); servo2.write(PWM2_MIN); delay(800);
  servo1.write(PWM1_MAX); servo2.write(PWM2_MAX); delay(800);
  servo1.write(S1_HOME);  servo2.write(S2_HOME);  delay(500);
  Serial.println(">> CALIBRATION COMPLETE.");
}

// ─────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────
void setup() {
  pinMode(EN_PIN, OUTPUT); digitalWrite(EN_PIN, HIGH);
  pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW);
  for(int i=0; i<4; i++) { pinMode(VALVE_PIN[i], OUTPUT); digitalWrite(VALVE_PIN[i], HIGH); }

  TS4::begin();
  stepper.setMaxSpeed(600);
  stepper.setAcceleration(300);

  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);

  Serial.begin(115200);
  RPI_SERIAL.begin(RPI_BAUD);
  RPI_SERIAL.setTimeout(10); 

  Wire.begin(); myGNSS.begin();
  Wire1.begin();
  if (oled.begin(OLED_ADDR, true)) { oledOK = true; oledStatus("READY"); }

  Serial5.begin(SC_BAUD_RATE, SERIAL_8N1);
  scServo.pSerial = &Serial5;

  calibratePWM();
  dualPrintf("SYSTEM ONLINE: RPI UART ACTIVE\n");
}

// ─────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────
void loop() {
  // 1. Polling GNSS
  if (gnssTimer >= 500) {
    gnssTimer = 0;
    if (myGNSS.getPVT()) {
      gnss.valid = true;
      gnss.lat = myGNSS.getLatitude() / 1e7;
      gnss.lon = myGNSS.getLongitude() / 1e7;
      gnss.sats = myGNSS.getSIV();
    }
  }

  // 2. Stepper Management & Tank Deduction
  if (stepper.isMoving) {
    stepperIdleTimer = 0;
  } else {
    if (stepperIdleTimer >= 100) digitalWrite(EN_PIN, HIGH);
    
    if (dispensing) {
      if (activeValveIdx >= 0) {
        tankLevel[activeValveIdx] -= currentDispenseML;
        if (tankLevel[activeValveIdx] < 0) tankLevel[activeValveIdx] = 0;
      }
      dispensing = false;
      for(int i=0; i<4; i++) { digitalWrite(VALVE_PIN[i], HIGH); valveState[i] = false; }
      
      bool low = false;
      for(int i=0; i<4; i++) if(tankLevel[i] < LOW_TANK_ML) low = true;
      digitalWrite(BUZZER_PIN, low ? HIGH : LOW);
      if(low) oledStatus("LOW TANK!");
      
      dualPrintf("[DISPENSE_END] Tank %s: %.1fml left\n", TANK_NAMES[activeValveIdx], tankLevel[activeValveIdx]);
    }
  }

  // 3. Serial Communication (RPi Priority)
  if (RPI_SERIAL.available()) {
    String input = RPI_SERIAL.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) processCommand(input);
  }
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) processCommand(input);
  }
}

// ─────────────────────────────────────────────────────────────
// COMMAND PROCESSOR
// ─────────────────────────────────────────────────────────────
void processCommand(String raw) {
  if (raw.startsWith("refill")) {
    for(int i=0; i<4; i++) tankLevel[i] = TANK_CAPACITY_ML;
    digitalWrite(BUZZER_PIN, LOW);
    oledStatus("REFILLED");
    return;
  }

  String tok[10];
  int count = 0, pos = 0;
  while (pos < (int)raw.length() && count < 10) {
    while (pos < (int)raw.length() && raw[pos] == ' ') pos++;
    if (pos >= (int)raw.length()) break;
    int end = pos;
    while (end < (int)raw.length() && raw[end] != ' ') end++;
    tok[count++] = raw.substring(pos, end);
    pos = end;
  }
  
  // OLED text extraction (field 11+)
  while (pos < (int)raw.length() && raw[pos] == ' ') pos++;
  String oledText = (pos < (int)raw.length()) ? raw.substring(pos) : "";

  if (count < 10) return;

  // PWM Servos
  if (tok[0] != "-") servo1.write(lastS1 = constrain(tok[0].toInt(), PWM1_MIN, PWM1_MAX));
  if (tok[1] != "-") servo2.write(lastS2 = constrain(tok[1].toInt(), PWM2_MIN, PWM2_MAX));

  // SC Smart Servos (Hard Safety Clamping 30-100)
  if (tok[2] != "-" || tok[3] != "-") {
    int a1_req = (tok[2] != "-") ? tok[2].toInt() : lastScAngle[0];
    int a2_req = (tok[3] != "-") ? tok[3].toInt() : lastScAngle[1];
    
    int a1 = constrain(a1_req, SC_MIN, SC_MAX);
    int a2 = constrain(a2_req, SC_MIN, SC_MAX);
    
    s16 posArr[2] = {(s16)map(a1, 0, 360, 0, 4095), (s16)map(a2, 0, 360, 0, 4095)};
    u16 speedArr[2] = {1500, 1500};
    byte accArr[2] = {50, 50};
    byte ids[2] = {0, 1};
    scServo.SyncWritePosEx(ids, 2, posArr, speedArr, accArr);
    lastScAngle[0] = a1; lastScAngle[1] = a2;
  }

  // Valves
  activeValveIdx = -1;
  for (int i = 0; i < 4; i++) {
    if (tok[5 + i] != "-") {
      valveState[i] = (tok[5 + i] == "on" || tok[5 + i] == "1");
      digitalWrite(VALVE_PIN[i], valveState[i] ? LOW : HIGH);
      if (valveState[i]) activeValveIdx = i;
    }
  }

  // Stepper
  if (tok[4] != "-") {
    long steps = tok[4].toInt();
    if (steps != 0) {
      if (activeValveIdx != -1 && tankLevel[activeValveIdx] <= 0) {
        dualPrintf("BLOCKED: TANK %s EMPTY\n", TANK_NAMES[activeValveIdx]);
        for(int i=0; i<4; i++) digitalWrite(VALVE_PIN[i], HIGH);
      } else {
        dispensing = true;
        currentDispenseML = abs(steps) / PULSES_PER_ML;
        digitalWrite(EN_PIN, LOW);
        stepper.moveRelAsync(steps);
      }
    }
  }

  if (oledText.length() > 0 && oledText != "-") oledStatus(oledText);
  
  // Handshake back to RPi
  dualPrintf("ACK:%d,%d,%ld,%.1f\n", lastS1, lastS2, totalSteps, (activeValveIdx != -1 ? tankLevel[activeValveIdx] : 0));
}