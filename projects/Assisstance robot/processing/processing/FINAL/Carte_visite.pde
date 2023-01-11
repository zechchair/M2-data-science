class carte extends PApplet {
 public boolean work2 = false;
ControlP5 cpExtra2;
PApplet parent;
 
carte(PApplet app2) {
super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this);
 
cpExtra2 = new ControlP5(this);
 
parent = app2;}
 
void settings() {
fullScreen();

 }
 void setup()
 {


String path="C:/Users/zechc/OneDrive - Université Mohammed VI Polytechnique/cours/CI2/mecatronique/all programs/all-in-one/processing/FINAL/"+list[1]+".jpg";
carte_image = loadImage(path,"jpg");

  image(background_cart, 0, 0,width,height);
  image(carte_image,780,300,400,400);


textFont(f,40);   
fill(#2399a3);
  text("Nom:",90,360);
  text(list[2],200,360);
  text("Prénom:",90,450);
  text(list[1],250,450);
  text("Fonction:",90,540);
  text(list[3],280,540);
  text("Bloc:",90,630);
    text(list[5],200,630);
  text("Adresse mail:",390,950);
  text(list[4],660,950);
  textFont(f1,100);
   fill(#2399a3);
   text(list[1]+" "+list[2],450,180);
   

}


void draw(){

}
public void show() {
 work2 = true;
 surface.setVisible(true);
 }
 public void hide() {
 work2 = false;
 surface.setVisible(false);}
}
