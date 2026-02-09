const int LDRPin = A0;
int reading = 0;

void setup() {
  pinMode(LDRPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  delay(100);
  reading = analogRead(LDRPin);
  Serial.println(reading);
}
