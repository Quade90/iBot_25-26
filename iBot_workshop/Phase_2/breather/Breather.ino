int pinLED = 3;

void setup()
{
  pinMode(pinLED, OUTPUT);
}

int flip = 1;
int i = 0;
  
void loop()
{
  if(flip == 1){
  	analogWrite(pinLED, i);
  	delay(10);
    if(i == 255){
      flip = 0;
    }
    else{
      i++;
    }
    
  }
  else{
  	analogWrite(pinLED, i);
    delay(10);
    if(i == 0){
      flip = 1;
    }
    else{
      i--;
    }
  }
}
