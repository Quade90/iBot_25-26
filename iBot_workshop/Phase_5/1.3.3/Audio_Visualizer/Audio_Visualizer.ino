#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1); 

const int soundPin = A0;
int sound = 0;
int height = 0;

void setup() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)){
    for(;;);
  }
  display.clearDisplay();
  pinMode(soundPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  display.clearDisplay();
  sound = analogRead(soundPin);
  height = map(sound, 400, 700, 0, 120);
  display.fillRect(0, 18, height, 30, WHITE);
  display.display();
  Serial.println(sound);
  delay(25);
}
