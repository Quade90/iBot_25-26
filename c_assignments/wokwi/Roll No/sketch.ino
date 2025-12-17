int pinLED = 12;

void setup()
{
  pinMode(pinLED, OUTPUT);
}

void loop()
{
  digitalWrite(pinLED, HIGH);
  delay(1000/30.0);
  digitalWrite(pinLED, LOW);
  delay(1000/30.0);
}