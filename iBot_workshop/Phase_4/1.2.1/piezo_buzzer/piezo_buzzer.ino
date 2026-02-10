const int piezoPin = 3;

void setup()
{
  pinMode(piezoPin, OUTPUT);
}

void loop()
{
  tone(piezoPin, 500);
  delay(1000);
  noTone(piezoPin);
  delay(1000);
}