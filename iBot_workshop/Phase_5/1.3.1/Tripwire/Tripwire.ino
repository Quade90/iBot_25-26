#include <LiquidCrystal.h>

const int LDRPin = A0;
const int buzzerPin = 8;
int reading = 0;

LiquidCrystal lcd_1(12, 11, 5, 4, 3, 2);

void setup()
{
  lcd_1.begin(16, 2);
  pinMode(LDRPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  Serial.begin(9600);
}

void loop(){
  reading = analogRead(LDRPin);
  Serial.println(reading);
  if(reading > 512){
    tone(buzzerPin, 500);
    lcd_1.clear();
    lcd_1.setCursor(0, 0);
    lcd_1.print("Security");
    lcd_1.setCursor(0, 1);
    lcd_1.print("Alert");
  }
  else{
    noTone(buzzerPin);
    lcd_1.clear();
    lcd_1.setCursor(0, 0);
    lcd_1.print("All");
    lcd_1.setCursor(0, 1);
    lcd_1.print("Good");
  }
  
}