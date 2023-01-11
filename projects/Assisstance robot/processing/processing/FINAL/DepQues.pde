 
class DepQ extends PApplet {
 public boolean work = false;
ControlP5 cpExtra1;
PApplet parent;
 
DepQ(PApplet app1) {
super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this);
 
cpExtra1 = new ControlP5(this);
 
parent = app1;


 PFont font = createFont("Times New Roman", 50);
 cpExtra1.addButton("Deplacement")
.setPosition(1000,500)
.setSize(700,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(#FFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra1.addButton("Question")
.setPosition(300,500)
.setSize(500,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(#FFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra1.addButton("Quitter")
.setPosition(650,900)
.setSize(500,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(#FFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
}
 
void settings() {
fullScreen();
}
 


void draw(){

background(255);
 texts();
}
 void texts(){
 


 text("Robot d'assistance",700,100);
textSize(50);

fill(#1A0099);
}

void Deplacement(){
 
myClient.write("deplacement");
d_q.hide();
 
}// send whatever you need to send here}

void Question(){
 myClient.write("question");
 d_q.hide();
}
void Quitter(){
 myClient.write("quitter");
 d_q.hide();
}
void mouseReleased() {

}
public void show() {
 work = true;
 surface.setVisible(true);
 }
 public void hide() {
 work = false;
 surface.setVisible(false);
 }}  
 
