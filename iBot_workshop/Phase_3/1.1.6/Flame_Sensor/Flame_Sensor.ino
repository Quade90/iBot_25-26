const int IRReceiverPin = A0;
int reading = 0;

void setup() {
  pinMode(IRReceiverPin, INPUT);
  Serial.begin(9600);
}

void loop() {
  reading = analogRead(IRReceiverPin);
  Serial.println(reading);  
  delay(100);
}
