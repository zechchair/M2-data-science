//Imports d_q
import processing.net.*; 
Client myClient;
import controlP5.*;
import uibooster.*;
import uibooster.model.*;
import uibooster.components.*; 
import java.util.Scanner;
import processing.opengl.*;
UiBooster booster;
PImage r2d2;
String in;
PImage carte_image;
PImage background_cart;
PImage emineslogo2;
PImage thanks;
PImage without_mouth;
PImage mask_required;
PImage raise_hand;
PImage pink_eyes;
PImage carte_visite;
PFont f_g;
PFont f;
PFont f1;
PFont f_a;
int maxImages = 8;
boolean mouthOpen = false;
PImage[] images = new PImage[maxImages];
int imageIndex = 0; // Initial image to be displayed is the first
boolean speaking = false;
int mouseClicks;

String[] list;

PWindow language;
DepQ d_q;
carte cart;
ControlP5 cp5;
Splashscreen splash1;


public void settings() {
 fullScreen();
}
 
void setup() {
myClient = new Client(this, "127.0.0.1", 8080); 
r2d2 =loadImage("r2d2.jpg");
emineslogo2=loadImage("emineslogo2.png");
without_mouth = loadImage("without_mouth.png");
background_cart = loadImage("fond.jpg");
pink_eyes = loadImage("pink_eyes.png");
thanks = loadImage("thanks.png");
raise_hand=loadImage("raise_hand.jpg");
mask_required=loadImage("mask_required.jpg");
carte_visite=loadImage("carte_visite.png");
f_g = createFont("Georgia", 40);
f_a = createFont("Arial", 40); 
f = loadFont("PalatinoLinotype-Bold-15.vlw");
f1 = createFont("Georgia", 40);
image(pink_eyes,0,0,width,height);
cp5 = new ControlP5(this); 
booster = new UiBooster();
for (int i = 0; i < 8; i ++ ) {
images[i] = loadImage(  i+".png" );
}
frameRate(7);
}
 
void draw() {
if (myClient.available() > 0) {
in = myClient.readString(); 
list = split(in, ',');
myClient.write("rcvd");

print("0 "); println(list[0]);
print("1 "); println(list[1]);
}
if (list[0].equals("nocart")){
  if (list[1].equals("choix")){d_q = new DepQ(this); list[1]="nono";}
  else if (list[1].equals("masktrue")){image(thanks,0,0,width,height);}
  else if (list[1].equals("speaktrue")){mouthOpen = true;
  mouth();}
  else if (list[1].equals("speakfalse")){mouthOpen = false;
  mouth();}
  else if(list[1].equals("hand")){image(raise_hand,0,0,width,height);}
  else if(list[1].equals("maskfalse")){image(mask_required,0,0,width,height);}

}
else if(list[0].equals("cart")){cart= new carte(this);delay(8000);cart.hide();}

}


void mousePressed() {
 if (mouseButton == LEFT) { mouseClicks++; } else { mouseClicks = 0; }
if(mouseClicks==1){
language = new PWindow(this);}}




 
 
void mouth(){
if (mouthOpen && frameCount % 40>0 && frameCount % 40 <40){ 
 image(without_mouth, 0, 0,width,height);
 image(images[imageIndex], 620, 670,700,600);
 imageIndex = int(random(images.length));}
else{ // if mouthOpen is false, mouth closes
 image(pink_eyes, 0, 0,width,height);
 
 }
}
