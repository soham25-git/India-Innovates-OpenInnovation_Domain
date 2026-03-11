/**
 * KRISHI-EYE Teensy 4.1 Firmware — Precision Spray Controller
 * 
 * Receives 10-field commands from Raspberry Pi via UART:
 *   <pwm1> <pwm2> <sc1> <sc2> <steps> <v1> <v2> <v3> <v4> <buzzer>
 * 
 * pwm1/pwm2 = Camera servos (horizontal/vertical)
 * sc1/sc2   = Nozzle smart servos (ST3215 via SMS_STS)
 * steps     = Stepper pump pulses (160 pulses/mL calibration)
 * v1-v4     = Solenoid valves (pins 26-29, active LOW)
 * buzzer    = Reserved (Teensy manages buzzer for tank warnings)
 * 
 * Tank Level System:
 *   - Tracks remaining fluid per tank (4 tanks for 4 valves)
 *   - Buzzer rings when any tank drops below LOW_TANK_ML
 *   - OLED shows which tank(s) are low
 *   - Auto-closes valves after stepper completes dispensing
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

// Servo limits
#define PWM1_MIN   30
#define PWM1_MAX  150
#define PWM2_MIN   10
#define PWM2_MAX   80
#define S1_HOME    90
#define S2_HOME    45

// SC Smart Servos — PHYSICAL SAFETY LIMITS
#define SC_MIN     30
#define SC_MAX    150

// Pump calibration: 20000 pulses = 125 mL
#define PULSES_PER_ML  160

// Auto-close valve delay after stepper finishes (ms)
#define VALVE_AUTO_CLOSE_MS 500

// ─────────────────────────────────────────────────────────────
// TANK LEVEL SYSTEM
// ─────────────────────────────────────────────────────────────
#define TANK_CAPACITY_ML  100.0   // Starting capacity per tank (mL)
#define LOW_TANK_ML        20.0    // Buzzer + OLED warning at 20mL remaining

const char* TANK_NAMES[NUM_VALVES] = {
  "BACT",   // Tank 1 (valve 1, pin 26) — Bacteria treatment
  "FUNG",   // Tank 2 (valve 2, pin 27) — Fungi treatment
  "NEMA",   // Tank 3 (valve 3, pin 28) — Nematode treatment
  "PEST"    // Tank 4 (valve 4, pin 29) — Pest treatment
};

float tankLevel[NUM_VALVES] = {
  TANK_CAPACITY_ML, TANK_CAPACITY_ML,
  TANK_CAPACITY_ML, TANK_CAPACITY_ML
};

// ─────────────────────────────────────────────────────────────
// OBJECTS & STATE
// ─────────────────────────────────────────────────────────────
SFE_UBLOX_GNSS    myGNSS;
PWMServo          servo1, servo2;
Stepper           stepper(STEP_PIN, DIR_PIN);
SMS_STS           scServo;
Adafruit_SH1106G  oled(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire1, -1);

bool oledOK = false;

const byte SC_ID[SC_NUM_SERVOS]  = {0, 1};
s16  scPos[SC_NUM_SERVOS]        = {2048, 2048};
u16  scSpeed[SC_NUM_SERVOS]      = {1500, 1500};
byte scACC[SC_NUM_SERVOS]        = {50, 50};

struct ScFeedback {
  bool  valid    = false;
  int   position = 0;
  float voltage  = 0;
  int   temp     = 0;
} scFb[SC_NUM_SERVOS];

int  lastServo1Angle            = S1_HOME;
int  lastServo2Angle            = S2_HOME;
int  lastScAngle[SC_NUM_SERVOS] = {-1, -1};
long totalStepsPulsed           = 0;
long lastMoveSteps              = 0;
bool valveState[NUM_VALVES]     = {false, false, false, false};
bool buzzerState                = false;

// Dispensing state
bool dispensing        = false;
int  dispenseValveIdx  = -1;
float dispensedML      = 0;

#define OLED_LINES 3
String oledLines[OLED_LINES] = {"", "", ""};

struct {
  bool   valid = false;
  int    hour, minute, second;
  double lat, lon;
  int    sats, fixType;
} gnss;

elapsedMillis gnssTimer, scFeedbackTimer, stepperIdleTimer;
const uint32_t GNSS_POLL_MS        = 500;
const uint32_t SC_FEEDBACK_WAIT_MS = 2000;
const uint32_t STEPPER_IDLE_MS     = 100;

bool scFeedbackPending = false;
bool stepperWasMoving  = false;

// Spray counters
unsigned long totalSprays = 0;
float totalMLDispensed    = 0;

// ─────────────────────────────────────────────────────────────
// DUAL OUTPUT
// ─────────────────────────────────────────────────────────────
void dualPrintln(const char* msg) {
  Serial.println(msg);
  RPI_SERIAL.println(msg);
}

void dualPrintf(const char* fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Serial.print(buf);
  RPI_SERIAL.print(buf);
}

// ─────────────────────────────────────────────────────────────
// OLED
// ─────────────────────────────────────────────────────────────
void oledRedraw() {
  if (!oledOK) return;
  oled.clearDisplay();
  oled.fillRect(0, 0, 128, 13, SH110X_WHITE);
  oled.setTextSize(1);
  oled.setTextColor(SH110X_BLACK);
  oled.setCursor(4, 3);
  oled.print("KRISHI-EYE CTRL");
  oled.setTextColor(SH110X_WHITE);
  oled.setTextSize(2);
  int yPos[OLED_LINES] = {16, 32, 48};
  for (int i = 0; i < OLED_LINES; i++) {
    if (oledLines[i].length() > 0) {
      oled.setCursor(0, yPos[i]);
      oled.print(oledLines[i].substring(0, 10));
    }
  }
  oled.display();
}

void oledPush(String text) {
  if (!oledOK) return;
  for (int i = 0; i < OLED_LINES - 1; i++) oledLines[i] = oledLines[i + 1];
  oledLines[OLED_LINES - 1] = text;
  oledRedraw();
}

// Show ONLY low/empty tanks on OLED
void oledShowTankStatus() {
  if (!oledOK) return;
  oled.clearDisplay();

  // Header
  oled.fillRect(0, 0, 128, 13, SH110X_WHITE);
  oled.setTextSize(1);
  oled.setTextColor(SH110X_BLACK);
  oled.setCursor(16, 3);
  oled.print("LOW TANK!");
  oled.setTextColor(SH110X_WHITE);

  int row = 0;
  for (int i = 0; i < NUM_VALVES; i++) {
    if (tankLevel[i] >= LOW_TANK_ML) continue;  // skip OK tanks

    int y = 16 + row * 16;
    oled.setTextSize(2);
    oled.setCursor(0, y);

    char line[16];
    if (tankLevel[i] <= 0) {
      snprintf(line, sizeof(line), "%s EMPTY", TANK_NAMES[i]);
    } else {
      snprintf(line, sizeof(line), "%s %dml", TANK_NAMES[i], (int)tankLevel[i]);
    }
    oled.print(line);
    row++;
    if (row >= 3) break;  // max 3 rows on screen
  }
  oled.display();
}

// ─────────────────────────────────────────────────────────────
// STEPPER / VALVE HELPERS
// ─────────────────────────────────────────────────────────────
inline void stepperEnable()  { digitalWrite(EN_PIN, LOW);  }
inline void stepperDisable() { digitalWrite(EN_PIN, HIGH); }

void closeAllValves() {
  for (int i = 0; i < NUM_VALVES; i++) {
    valveState[i] = false;
    digitalWrite(VALVE_PIN[i], HIGH);
  }
}

// ─────────────────────────────────────────────────────────────
// TANK WARNING CHECK — called after every dispense
// ─────────────────────────────────────────────────────────────
void checkTankLevels() {
  bool anyLow = false;
  String lowTanks = "";

  for (int i = 0; i < NUM_VALVES; i++) {
    if (tankLevel[i] < LOW_TANK_ML) {
      anyLow = true;
      if (lowTanks.length() > 0) lowTanks += ",";
      lowTanks += TANK_NAMES[i];

      if (tankLevel[i] <= 0) {
        dualPrintf("  [ALERT] Tank %s is EMPTY!\n", TANK_NAMES[i]);
      } else {
        dualPrintf("  [WARNING] Tank %s LOW: %.0f mL remaining\n",
                   TANK_NAMES[i], tankLevel[i]);
      }
    }
  }

  if (anyLow) {
    // Buzzer ON — continuous warning
    buzzerState = true;
    digitalWrite(BUZZER_PIN, HIGH);

    // Show on OLED which tanks are low
    oledShowTankStatus();
  } else {
    // All tanks OK — buzzer OFF
    buzzerState = false;
    digitalWrite(BUZZER_PIN, LOW);
  }
}

// ─────────────────────────────────────────────────────────────
// CALIBRATION
// ─────────────────────────────────────────────────────────────
void calibratePWM() {
  dualPrintln(">> INITIATING CALIBRATION SWEEP...");

  servo1.write(PWM1_MIN);
  servo2.write(PWM2_MIN);
  delay(1000);

  servo1.write(PWM1_MAX);
  servo2.write(PWM2_MAX);
  delay(1000);

  servo1.write(S1_HOME);
  servo2.write(S2_HOME);
  lastServo1Angle = S1_HOME;
  lastServo2Angle = S2_HOME;
  delay(800);

  dualPrintln(">> CALIBRATION COMPLETE.");
}

// ─────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────
void setup() {
  pinMode(EN_PIN, OUTPUT);     stepperDisable();
  pinMode(STEP_PIN, OUTPUT);   digitalWrite(STEP_PIN, LOW);
  pinMode(DIR_PIN, OUTPUT);    digitalWrite(DIR_PIN, LOW);
  pinMode(BUZZER_PIN, OUTPUT); digitalWrite(BUZZER_PIN, LOW);

  TS4::begin();
  stepper.setMaxSpeed(600);
  stepper.setAcceleration(300);

  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);

  Serial.begin(115200);
  uint32_t t = millis();
  while (!Serial && (millis() - t < 3000));

  RPI_SERIAL.begin(RPI_BAUD);

  Wire.begin();
  Wire.setClock(400000);
  myGNSS.begin();

  Wire1.begin();
  if (oled.begin(OLED_ADDR, true)) {
    oledOK = true;
    oled.clearDisplay();
    oledShowTankStatus();  // Show initial tank levels
  }

  for (int i = 0; i < NUM_VALVES; i++) {
    pinMode(VALVE_PIN[i], OUTPUT);
    digitalWrite(VALVE_PIN[i], HIGH);
  }

  Serial5.begin(SC_BAUD_RATE, SERIAL_8N1);
  scServo.pSerial = &Serial5;

  calibratePWM();

  dualPrintln("KRISHI-EYE TEENSY 4.1 READY");
  dualPrintf("Tanks: %.0f mL each | Warning at: %.0f mL\n",
             TANK_CAPACITY_ML, LOW_TANK_ML);
  printHelp();
  Serial.print("\nCMD> ");
  RPI_SERIAL.print("\nCMD> ");
}

// ─────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────
void loop() {
  // GNSS polling
  if (gnssTimer >= GNSS_POLL_MS) {
    gnssTimer = 0;
    pollGNSS();
  }

  // SC servo feedback
  if (scFeedbackPending && scFeedbackTimer >= SC_FEEDBACK_WAIT_MS) {
    scFeedbackPending = false;
    readScFeedback();
  }

  // Stepper idle disable
  if (stepper.isMoving) {
    stepperWasMoving = true;
    stepperIdleTimer = 0;
  } else if (stepperWasMoving) {
    if (stepperIdleTimer >= STEPPER_IDLE_MS) {
      stepperWasMoving = false;
      stepperDisable();
    }
  }

  // AUTO-CLOSE valves + deduct tank after dispensing completes
  if (dispensing && !stepper.isMoving) {
    delay(VALVE_AUTO_CLOSE_MS);
    closeAllValves();
    dispensing = false;
    totalSprays++;
    totalMLDispensed += dispensedML;

    // Deduct from the correct tank
    if (dispenseValveIdx >= 0 && dispenseValveIdx < NUM_VALVES) {
      tankLevel[dispenseValveIdx] -= dispensedML;
      if (tankLevel[dispenseValveIdx] < 0) tankLevel[dispenseValveIdx] = 0;
    }

    // Send feedback to RPi
    dualPrintln("\n[SPRAY COMPLETE]");
    dualPrintf("  Dispensed: %.1f mL from tank %s\n",
               dispensedML, 
               (dispenseValveIdx >= 0 ? TANK_NAMES[dispenseValveIdx] : "?"));
    dualPrintf("  Tank %s remaining: %.0f mL\n",
               (dispenseValveIdx >= 0 ? TANK_NAMES[dispenseValveIdx] : "?"),
               (dispenseValveIdx >= 0 ? tankLevel[dispenseValveIdx] : 0));
    dualPrintf("  Total sprays: %lu | Total dispensed: %.1f mL\n",
               totalSprays, totalMLDispensed);

    // Check tank levels → buzzer + OLED if low
    checkTankLevels();

    // If no tank warning, show normal OLED
    bool anyLow = false;
    for (int i = 0; i < NUM_VALVES; i++) {
      if (tankLevel[i] < LOW_TANK_ML) { anyLow = true; break; }
    }
    if (!anyLow) {
      oledPush(String(dispensedML, 0) + "mL OK");
    }

    printFullReport();
    Serial.print("\nCMD> ");
    RPI_SERIAL.print("\nCMD> ");
  }

  // USB Serial commands
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      processCommand(input);
      Serial.print("\nCMD> ");
      RPI_SERIAL.print("\nCMD> ");
    }
  }

  // RPi Serial commands
  if (RPI_SERIAL.available()) {
    String input = RPI_SERIAL.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      processCommand(input);
      Serial.print("\nCMD> ");
      RPI_SERIAL.print("\nCMD> ");
    }
  }
}

// ─────────────────────────────────────────────────────────────
// COMMAND PROCESSOR
// ─────────────────────────────────────────────────────────────
void processCommand(String raw) {
  String lower = raw;
  lower.toLowerCase();

  if (lower == "help")   { printHelp();       return; }
  if (lower == "status") { printFullReport(); return; }
  if (lower == "tanks")  { printTankStatus(); oledShowTankStatus(); return; }
  if (lower.startsWith("refill")) {
    handleRefill(lower);
    return;
  }

  // Tokenize
  String tok[10];
  int count = 0, pos = 0, len = raw.length();

  while (pos < len && count < 10) {
    while (pos < len && raw[pos] == ' ') pos++;
    if (pos >= len) break;
    int end = pos;
    while (end < len && raw[end] != ' ') end++;
    tok[count++] = raw.substring(pos, end);
    pos = end;
  }

  if (count < 10) {
    dualPrintf("  [ERROR] Need 10 fields, got %d.\n", count);
    return;
  }

  dualPrintln("\n[ACK]");

  // ── PWM Servos (camera) ─────────────────────────────────
  if (tok[0] != "-") {
    int r = tok[0].toInt();
    lastServo1Angle = constrain(r, PWM1_MIN, PWM1_MAX);
    servo1.write(lastServo1Angle);
  }
  if (tok[1] != "-") {
    int r = tok[1].toInt();
    lastServo2Angle = constrain(r, PWM2_MIN, PWM2_MAX);
    servo2.write(lastServo2Angle);
  }

  // ── SC Smart Servos (nozzle) ────────────────────────────
  if (tok[2] != "-" || tok[3] != "-") {
    int ang1_req = (tok[2] != "-") ? tok[2].toInt() : (lastScAngle[0] >= 0 ? lastScAngle[0] : SC_MIN);
    int ang2_req = (tok[3] != "-") ? tok[3].toInt() : (lastScAngle[1] >= 0 ? lastScAngle[1] : SC_MIN);

    int ang1 = constrain(ang1_req, SC_MIN, SC_MAX);
    int ang2 = constrain(ang2_req, SC_MIN, SC_MAX);

    if (ang1_req != ang1) dualPrintf("  [SAFETY] sc1 %d -> %d clamped\n", ang1_req, ang1);
    if (ang2_req != ang2) dualPrintf("  [SAFETY] sc2 %d -> %d clamped\n", ang2_req, ang2);

    scPos[0] = (s16)map(ang1, 0, 360, 0, 4095);
    scPos[1] = (s16)map(ang2, 0, 360, 0, 4095);
    lastScAngle[0] = ang1;
    lastScAngle[1] = ang2;
    scServo.SyncWritePosEx((byte*)SC_ID, SC_NUM_SERVOS, scPos, scSpeed, scACC);
    // Read feedback synchronously (200ms for servos to settle)
    delay(200);
    readScFeedback();
    dualPrintf("  Nozzle H: %d deg | V: %d deg\n", ang1, ang2);
    for (int i = 0; i < SC_NUM_SERVOS; i++) {
      if (scFb[i].valid)
        dualPrintf("  SC%d fb: pos=%d | %.1fV | %dC\n",
                   i, scFb[i].position, scFb[i].voltage, scFb[i].temp);
    }
  }

  // ── Valves ──────────────────────────────────────────────
  dispenseValveIdx = -1;
  for (int i = 0; i < NUM_VALVES; i++) {
    if (tok[5 + i] != "-") {
      valveState[i] = (tok[5 + i] == "on" || tok[5 + i] == "1");
      digitalWrite(VALVE_PIN[i], valveState[i] ? LOW : HIGH);
      if (valveState[i]) dispenseValveIdx = i;
    }
  }

  // ── Check if the target tank has enough fluid ───────────
  if (dispenseValveIdx >= 0 && tankLevel[dispenseValveIdx] <= 0) {
    dualPrintf("  [BLOCKED] Tank %s is EMPTY — cannot spray!\n",
               TANK_NAMES[dispenseValveIdx]);
    closeAllValves();
    checkTankLevels();
    return;
  }

  // ── Buzzer from packet (ignored — Teensy manages buzzer) ─
  // tok[9] is received but we don't use it for buzzer control

  // ── Stepper (pump) ──────────────────────────────────────
  if (tok[4] != "-") {
    long steps = tok[4].toInt();
    if (steps != 0) {
      lastMoveSteps = abs(steps);
      totalStepsPulsed += abs(steps);
      dispensedML = (float)abs(steps) / (float)PULSES_PER_ML;
      dispensing = true;

      dualPrintf("  Pump: %ld pulses (%.1f mL) from tank %s\n",
                 steps, dispensedML,
                 (dispenseValveIdx >= 0 ? TANK_NAMES[dispenseValveIdx] : "?"));

      stepperEnable();
      stepper.moveRelAsync(steps);
    } else {
      stepper.stop();
      stepperDisable();
      dispensing = false;
    }
  }

  // If no dispensing, send immediate ACK with full status
  if (!dispensing) {
    // Include GNSS in ACK
    if (gnss.valid)
      dualPrintf("  GNSS: %.6f, %.6f | %d sats\n", gnss.lat, gnss.lon, gnss.sats);
    printFullReport();
  }
}

// ─────────────────────────────────────────────────────────────
// REFILL COMMAND — manually reset tank level
// Usage: "refill all" or "refill 1" or "refill 2" etc
// ─────────────────────────────────────────────────────────────
void handleRefill(String cmd) {
  if (cmd == "refill all") {
    for (int i = 0; i < NUM_VALVES; i++) tankLevel[i] = TANK_CAPACITY_ML;
    dualPrintln("  All tanks refilled.");
  } else {
    int idx = cmd.substring(7).toInt() - 1;  // "refill 1" → idx 0
    if (idx >= 0 && idx < NUM_VALVES) {
      tankLevel[idx] = TANK_CAPACITY_ML;
      dualPrintf("  Tank %s refilled to %.0f mL.\n",
                 TANK_NAMES[idx], TANK_CAPACITY_ML);
    } else {
      dualPrintln("  Usage: refill all | refill 1 | refill 2 | refill 3 | refill 4");
    }
  }
  // Clear buzzer if all tanks now OK
  checkTankLevels();
  oledShowTankStatus();
  printTankStatus();
}

// ─────────────────────────────────────────────────────────────
// GNSS
// ─────────────────────────────────────────────────────────────
void pollGNSS() {
  if (myGNSS.getPVT()) {
    gnss.valid    = true;
    gnss.lat      = myGNSS.getLatitude()  / 10000000.0;
    gnss.lon      = myGNSS.getLongitude() / 10000000.0;
    gnss.sats     = myGNSS.getSIV();
    gnss.fixType  = myGNSS.getFixType();
    gnss.hour     = myGNSS.getHour();
    gnss.minute   = myGNSS.getMinute();
    gnss.second   = myGNSS.getSecond();
  }
}

// ─────────────────────────────────────────────────────────────
// SC SERVO FEEDBACK
// ─────────────────────────────────────────────────────────────
void readScFeedback() {
  for (int i = 0; i < SC_NUM_SERVOS; i++) {
    int pos = scServo.ReadPos(SC_ID[i]);
    if (pos != -1) {
      scFb[i].valid    = true;
      scFb[i].position = pos;
      scFb[i].voltage  = (float)scServo.ReadVoltage(SC_ID[i]) / 10.0f;
      scFb[i].temp     = scServo.ReadTemper(SC_ID[i]);
    }
  }
}

// ─────────────────────────────────────────────────────────────
// STATUS REPORTS
// ─────────────────────────────────────────────────────────────
void printTankStatus() {
  dualPrintln("\n─── TANK LEVELS ───");
  for (int i = 0; i < NUM_VALVES; i++) {
    const char* status;
    if (tankLevel[i] <= 0) status = "EMPTY";
    else if (tankLevel[i] < LOW_TANK_ML) status = "LOW";
    else status = "OK";
    dualPrintf("  Tank %d (%s): %5.0f / %.0f mL [%s]\n",
               i + 1, TANK_NAMES[i], tankLevel[i], TANK_CAPACITY_ML, status);
  }
  dualPrintln("───────────────────");
}

void printFullReport() {
  dualPrintln("\n========== SYSTEM STATUS ==========");

  if (!gnss.valid)
    dualPrintln("  GNSS     : Waiting...");
  else
    dualPrintf("  GNSS     : %d sats | %.6f, %.6f\n",
               gnss.sats, gnss.lat, gnss.lon);

  dualPrintf("  Camera   : H=%d V=%d\n", lastServo1Angle, lastServo2Angle);

  for (int i = 0; i < SC_NUM_SERVOS; i++) {
    if (scFb[i].valid)
      dualPrintf("  Nozzle%d  : pos=%d | cmd=%d deg | %.1fV %dC\n",
                 i, scFb[i].position, lastScAngle[i],
                 scFb[i].voltage, scFb[i].temp);
    else
      dualPrintf("  Nozzle%d  : cmd=%d deg\n", i, lastScAngle[i]);
  }

  dualPrintf("  Pump     : %s | Total=%ld pulses\n",
             stepper.isMoving ? "RUNNING" : "IDLE", totalStepsPulsed);
  dualPrintf("  Valves   : %d %d %d %d\n",
             valveState[0], valveState[1], valveState[2], valveState[3]);
  dualPrintf("  Sprays   : %lu total | %.1f mL total\n",
             totalSprays, totalMLDispensed);

  // Tank summary line
  for (int i = 0; i < NUM_VALVES; i++) {
    dualPrintf("  Tank%d    : %.0f mL %s\n",
               i + 1, tankLevel[i],
               (tankLevel[i] <= 0 ? "[EMPTY]" :
                (tankLevel[i] < LOW_TANK_ML ? "[LOW!]" : "")));
  }

  dualPrintln("====================================");
}

void printHelp() {
  dualPrintln("KRISHI-EYE Commands:");
  dualPrintln("  <pwm1> <pwm2> <sc1> <sc2> <steps> <v1> <v2> <v3> <v4> <buz>");
  dualPrintln("  status  — Full system report");
  dualPrintln("  tanks   — Tank levels");
  dualPrintln("  refill all | refill 1-4 — Reset tank level");
  dualPrintln("  help    — This message");
}
