const int buttonPin = 2;
const int pinLED = 4;
int currButtonState = 0;
int prevButtonState = 0;
bool buttonOn = false;

void setup()
{
  pinMode(buttonPin, INPUT);
  pinMode(pinLED, OUTPUT);
}

void loop()
{
  currButtonState = digitalRead(buttonPin);
  if(currButtonState == HIGH && currButtonState != prevButtonState){
  	buttonOn = !buttonOn;
  }
  if(buttonOn){
  	digitalWrite(pinLED, HIGH);
  }
  else{
  	digitalWrite(pinLED, LOW);
  }
  prevButtonState = currButtonState;
}