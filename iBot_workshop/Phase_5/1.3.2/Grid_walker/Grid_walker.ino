#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1); 

int posx = 64;
int posy = 32;
int upPin = 3;
int downPin = 4;
int leftPin = 5;
int rightPin = 6;
bool upPrev = 0;
bool downPrev = 0;
bool leftPrev = 0;
bool rightPrev = 0;
int up = 0;
int down = 0;
int left = 0;
int right = 0;

void setup() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)){
    for(;;);
  }
  display.clearDisplay();
  display.drawLine(127, 63, 0, 63, WHITE);
  display.drawLine(127, 0, 127, 63, WHITE);
  for(int i = 0; i<127; i+=8){
    display.drawLine(i, 0, i, 63, WHITE);
  }
  for(int i = 0; i<63; i+=8){
    display.drawLine(0, i, 127, i, WHITE);
  }
  display.fillRect(posx, posy, 8, 8, WHITE);

  display.display();

  pinMode(upPin, INPUT_PULLUP);
  pinMode(downPin, INPUT_PULLUP);
  pinMode(leftPin, INPUT_PULLUP);
  pinMode(rightPin, INPUT_PULLUP);

  Serial.begin(9600);
}

void loop() {
  up = digitalRead(upPin);
  down = digitalRead(downPin);
  left = digitalRead(leftPin);
  right = digitalRead(rightPin);

  if(up == LOW && up != upPrev){
    display.fillRect(posx+1, posy+1, 7, 7, BLACK);
    if(posy == 0){
      posy = 56;
    }
    else{
      posy-=8;
    }
    display.fillRect(posx, posy, 8, 8, WHITE);
    display.display();
    Serial.println("Up");
  }
  if(down == LOW && down != downPrev){
    display.fillRect(posx+1, posy+1, 7, 7, BLACK);
    if(posy == 56){
      posy = 0;
    }
    else{
      posy+=8;
    }
    display.fillRect(posx, posy, 8, 8, WHITE);
    display.display();
    Serial.println("Down");
  }
  if(left == LOW && left != leftPrev){
    display.fillRect(posx+1, posy+1, 7, 7, BLACK);
    if(posx == 0){
      posx = 120;
    }
    else{
      posx-=8;
    }
    display.fillRect(posx, posy, 8, 8, WHITE);
    display.display();
    Serial.println("Left");
  }
  if(right == LOW && right != rightPrev){
    display.fillRect(posx+1, posy+1, 7, 7, BLACK);
    if(posx == 120){
      posx = 0;
    }
    else{
      posx+=8;
    }
    display.fillRect(posx, posy, 8, 8, WHITE);
    display.display();
    Serial.println("Right");
  }

  upPrev = up;
  downPrev = down;
  leftPrev = left;
  rightPrev = right;

  display.drawLine(127, 63, 0, 63, WHITE);
  display.drawLine(127, 0, 127, 63, WHITE);
  display.display();  

  delay(100);
}
