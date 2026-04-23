/*
 * ExerSense - Bicep Curl Tracker
 * Reads two MPU6050 IMUs (forearm + elbow) and outputs CSV data for Python processing
 * 
 * Hardware:
 *   - Arduino Nano 33 IoT
 *   - MPU6050 on forearm (I2C addr 0x68)
 *   - MPU6050 on elbow (I2C addr 0x69, AD0 pulled HIGH)
 * 
 * Wiring:
 *   - SDA -> A4, SCL -> A5 (standard Arduino I2C pins)
 *   - Both MPUs share the same I2C bus
 */

#include <Wire.h>

// I2C addresses - 0x68 is default, 0x69 when AD0 pin is HIGH
#define FOREARM_IMU 0x68
#define ELBOW_IMU   0x69

// Accelerometer readings - using int16_t because MPU6050 outputs 16-bit signed values
int16_t forearmX, forearmY, forearmZ;
int16_t elbowX, elbowY, elbowZ;
int16_t elbowBaseX, elbowBaseY, elbowBaseZ;  // baseline captured at startup

int Y_TOP = 8000;
int Y_BOTTOM = -5000;

// State tracking
int reps = 0;
int atTop = 0;  
unsigned long t0;  // start time for relative timestamps


/*
 * Reads accelerometer XYZ from MPU6050 at given I2C address
 * MPU6050 accel data starts at register 0x3B, 6 bytes total (2 per axis)
 */
void readAccel(int addr, int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);  
  Wire.endTransmission(false);  
  Wire.requestFrom(addr, 6);  // 6 bytes: XH,XL,YH,YL,ZH,ZL
  

  x = Wire.read() << 8 | Wire.read();
  y = Wire.read() << 8 | Wire.read();
  z = Wire.read() << 8 | Wire.read();
}


/*
 * Wakes up MPU6050 from sleep mode
 * 
 * Register 0x6B is PWR_MGMT_1, writing 0x00 clears sleep bit
 * invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf (page 40)
 */
void wakeMPU(int addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0x00);  // clear sleep bit
  Wire.endTransmission();
}


void setup() {
  // 9600 baud for data streaming
  Serial.begin(9600);
  
  // Init I2C as master
  // pjrc.com/teensy/td_libs_Wire.html
  Wire.begin();
  
  // Wake both sensors
  wakeMPU(FOREARM_IMU);
  wakeMPU(ELBOW_IMU);
  
  // Let sensors settle before grabbing baseline
  delay(500);
  readAccel(ELBOW_IMU, elbowBaseX, elbowBaseY, elbowBaseZ);
  
  // Record start time for timestamps
  // arduino.cc/reference/en/language/functions/time/millis/
  t0 = millis();
  
  Serial.println("READY");
}


void loop() {
  // Read current sensor values
  readAccel(FOREARM_IMU, forearmX, forearmY, forearmZ);
  readAccel(ELBOW_IMU, elbowX, elbowY, elbowZ);
  
  // Calculate elbow drift from baseline
  int driftX = elbowX - elbowBaseX;
  int driftY = elbowY - elbowBaseY;
  int driftZ = elbowZ - elbowBaseZ;
  
  // Output CSV: forearmX,Y,Z,elbowDriftX,Y,Z,timestamp
  Serial.print("DATA,");
  Serial.print(forearmX); Serial.print(",");
  Serial.print(forearmY); Serial.print(",");
  Serial.print(forearmZ); Serial.print(",");
  Serial.print(driftX); Serial.print(",");
  Serial.print(driftY); Serial.print(",");
  Serial.print(driftZ); Serial.print(",");
  Serial.println(millis() - t0);
  
  // Transition: bottom -> top -> bottom = 1 rep
  if (!atTop && forearmY > Y_TOP) {
    atTop = 1;
  } 
  else if (atTop && forearmY < Y_BOTTOM) {
    atTop = 0;
    reps++;
    Serial.print("REP,");
    Serial.println(reps);
  }
  
  // 50ms delay = 20Hz sampling rate
  delay(50);
}