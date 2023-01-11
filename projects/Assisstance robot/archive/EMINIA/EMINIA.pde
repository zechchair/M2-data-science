  /*                                               Interface graphique Robot d'assistance EMINIA
                                         Cet interface comporte 1-Interactive face
                                                                2-Menu(Language-departement/question-carte visite)
                                                                3-Niveau de charge et message alerte au cas de déchargement 
*/
//Importations

import gifAnimation.*;
import processing.net.*; 
Client myClient;
import controlP5.*;
import java.util.Scanner;
import processing.opengl.*;
//Objects definitions
PWindow language;
DepQ d_q;
carte cart;
Gif myAnimation,myAnimation1,myAnimation2;
//Declaration
//1-Images
PImage carte_image,background_cart,background_principal,logo,thanks, without_mouth,mask_required,raise_hand,pink_eyes, charging,quit,battery;
int maxImages = 8;
PImage[] images = new PImage[maxImages];
int imageIndex = 0; // Initial image to be displayed is the firs
//2-Fonts
PFont f_g,f,f1,f_a,fontt,fontt1;
//Booleans
boolean mouthOpen = false,speaking = false;
//numeric variables
int mouseClicks;
float x,o,d;
int charge=100;
 String s;
String[] list;
//for quit button
int bX = 1750, bY = 900;
String  in;
public void settings() {
 fullScreen();
}
 
public void setup() {
  //Connection python
myClient = new Client(this, "127.0.0.1", 8080);
//Loads
myAnimation = new Gif(this, "Tongue_Big.gif");
myAnimation.play();
myAnimation1 = new Gif(this, "Sad_Big.gif");
myAnimation1.play();
myAnimation2 = new Gif(this, "Sleep_Big.gif");
myAnimation2.play();
background_principal=loadImage("BotBackground.jpg");  
logo=loadImage("logo.png");
battery=loadImage("battery.png");
battery.resize(100,100);
quit =loadImage("quit1.png");
 quit.resize(120,120);
without_mouth = loadImage("without_mouth.png");
background_cart = loadImage("fond.jpg");
pink_eyes = loadImage("pink_eyes.png");
thanks = loadImage("thanks.png");
raise_hand=loadImage("raise_hand.jpg");
mask_required=loadImage("mask_required.jpg");

f_g = createFont("Georgia", 40);
f_a = createFont("Arial", 40); 
f = loadFont("PalatinoLinotype-Bold-15.vlw");
f1 = createFont("Georgia", 40);
charging = loadImage("lighting.png");
fontt = loadFont("Arial-Black-48.vlw");
fontt1 = loadFont("Arial-BoldMT-48.vlw");
//Background face
image(pink_eyes,0,0,width,height);

for (int i = 0; i < 8; i ++ ) {
images[i] = loadImage(  i+".png" );
}
frameRate(7);
}
 
public void draw() {
   //If the serial is available read the list
if (myClient.available() > 0) {
in = myClient.readString(); 
 myClient.write("rcvd");
list = split(in, ',');

charge=Integer.valueOf(list[0]);
println(charge);
 println(list[1]);
 println(list[2]);

}
////conditions on the data that we had read

  if (list[1].equals("speak")){
    
 
    if (list[2].equals("speakfalse")){mouthOpen = false;mouth();}
  else if (list[2].equals("speaktrue")){mouthOpen = true;mouth();}
}
else if (list[1].equals("yellow")){d_q = new DepQ(this);list[1]="nami";}
else if (list[1].equals("lang")){language = new PWindow(this);list[1]="nami";}
  
else if(list[1].equals("hide")){d_q.hide();}
else if(list[1].equals("hidelang")){language.hide();}


else if(list[1].equals("cart")){cart= new carte(this);delay(8000);cart.hide();}


}
// if mouth pressed open language window
/*
void mousePressed() {
 if (mouseButton == LEFT) { mouseClicks++; } else { mouseClicks = 0; }
if(mouseClicks==1){
language = new PWindow(this);}}

*/


 
 //void for opening the mouth
void mouth(){
if (mouthOpen && frameCount % 40>0 && frameCount % 40 <40){ 
 image(without_mouth, 0, 0,width,height);
 charge();
 image(images[imageIndex], 620, 670,700,600);
 imageIndex = int(random(images.length));}
else{ // if mouthOpen is false, mouth closes
 
 image(pink_eyes, 0, 0,width,height);
 charge();
 
 
 }
}

//Void for determining the charge
//firs you can find in declaration in Eminia sketch that charge was set on a specific value that was read from python and vary every time
void charge(){
   // design of the icon charge
  ellipseMode(CENTER);
  fill(#ffc742);

  ellipse(x, o, d, d);//yellow1
  fill(#ffc742);
  ellipse(x, o, d-30, d-30);//green1
  fill(#ffc742);
   

  showArcs();
 //this part is just to make sure that charge coud vary
  // -----------------
  //Press + to add and -  to remove
  if (keyPressed&&key=='+') 
  
    charge++;
 
  if (keyPressed&&key=='-') 
    charge--;
 
 image(charging, 1780, 80,50,50);
  fill(255); // white
text_charge();

}
//text of percentage
void text_charge(){    
  textFont(fontt, 20);
 fill(#000000);
 text(charge+"%",1785,105);}
 
 
void showArcs(){
  if (charge<15){
   
   
  fill(#FF0000);   // !!!!!!!!!!!!!!!!!!!!!!!!!

  arc(x, o, d, d, PI+HALF_PI, map(charge, 0, 75, PI+HALF_PI, PI+HALF_PI+PI+HALF_PI));//yellow  !!!!!!!!!!!!!!!   

 
  fill(255);
  arc(x, o, d-30, d-30, 0, TWO_PI);//center
  //booster = new UiBooster();
  //booster.showWarningDialog("Batterie faible ... ", "Warning");

 // Conditions on battery level


}
  else{
     fill(#10E042);   // !!!!!!!!!!!!!!!!!!!!!!!!!
      
 
  arc(x, o, d, d, PI+HALF_PI, map(charge, 0, 75, PI+HALF_PI, PI+HALF_PI+PI+HALF_PI));//yellow  !!!!!!!!!!!!!!!   

 
  fill(255);
  arc(x, o, d-30, d-30, 0, TWO_PI);//center}
  
  }
 
}
//De
