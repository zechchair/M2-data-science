import pygame
import math
from queue import PriorityQueue
import time
RR=0
DD=0
UU=0
LL=0
RD=0
RU=0
LD=0
LU=0
direction ="R"
xs=1
ys=1
ii=xs
jj=ys
xf=350
yf=70
Commande=[]

start_time=0
to = 'nicolas'
pixels=[]
TempBarrier = []
vitesse = 60 #km per h
grid_width=0.002*3600 #grille en km


WIDTH = 1200
WIDTH2= 450
pix = 375 # nbr de grille

pixel_distance = 1
WIN = pygame.display.set_mode((WIDTH, WIDTH2))
#pix = 50# nbr de grille
#WIDTH = 800
#WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
def main(win, width):
	global run,Commande,start_time
	global TempBarrier
	ROWS = pix
	grid = make_grid(ROWS, width)

	start = None
	end = None
	cnt = False
	c=0

	run = True
	
	
	while run:
		obstacles(grid)
		end=FROM(grid)
		start=vers(grid)
		draw(win, grid, ROWS, width)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]: # LEFT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				spot = grid[row][col]
               
				if not start and spot != end:
					start = spot
					start.make_start()
				if spot != end and spot != start:
					spot.make_barrier()	
			elif pygame.mouse.get_pressed()[2]: # RIGHT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				spot = grid[row][col]
				spot.reset()
				if spot == start:
					start = None
				elif spot == end:
					end = None
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE and start and end:
					start_time=time.time()
					for row in grid:
						for spot in row:
							spot.update_neighbors(grid, 0)
					
					algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, width)
	 
		#if len(Commande)>2 and c<10 :
		#	c+=1
		#	print(Commande)
	pygame.quit()

def reconstruct_path(came_from, current, draw,grid):

	global pix, start_time, vitesse, grid_width, pixels,xf,yf,xs,ys,WIDTH,WIN,Commande
	count=0
	distance2 = 0
	


	while current in came_from  :

		#distance2 += h(current.get_pos(), came_from[current].get_pos(), 1)
		
		pixels.append([came_from[current].get_row(),current.get_row(),came_from[current].get_col(),current.get_col()])
		current = came_from[current]
		current.make_path()
		xs=current.get_row()
		ys=current.get_col()
		instruction1(count)
		
		count=count+1
	b=time.time()-start_time
	draw()
	instructionf()
	print(b)
	print(Commande)
	

	

def instructionf():
	global RR,DD,UU,LL,RD,RU,LD,LU
	if LD!=0:
		Commande.append(["LD",LD])
		LD=0
	if RU!=0:
		Commande.append(["RU",RU])
		RU=0
	if RD!=0:
		Commande.append(["RD",RD])
		RD=0
	if UU!=0:
		Commande.append(["UU",UU])
		UU=0
	if DD!=0:
		Commande.append(["DD",DD])
		DD=0
	if LL!=0:
		Commande.append(["LL",LL])
		LL=0

	if LU!=0:
		Commande.append(["LU",LU])
		LU=0
	if RR!=0:
		Commande.append(["RR",RR])
		RR=0

def instruction1(i):
	global  pixels,Commande,RR,DD,UU,LL,RD,RU,LD,LU,pixel_distance


	if  pixels[i][0]==pixels[i][1]+1 and pixels[i][2]==pixels[i][3]:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
	
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("RR")
		
		RR=RR+1*pixel_distance
	
	elif  pixels[i][0]==pixels[i][1]-1 and pixels[i][2]==pixels[i][3]:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("LL")
		LL=LL+1*pixel_distance
	
	
	elif  pixels[i][0]==pixels[i][1] and pixels[i][2]==pixels[i][3]+1:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("DD")
		DD=DD+1*pixel_distance
	

	elif pixels[i][0]==pixels[i][1] and pixels[i][2]==pixels[i][3]-1:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("UU")
		UU=UU+1*pixel_distance


	elif  pixels[i][0]==pixels[i][1]+1 and pixels[i][2]==pixels[i][3]+1:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("RD")
		RD=RD+1.4*pixel_distance

	elif pixels[i][0]==pixels[i][1]+1 and pixels[i][2]==pixels[i][3]-1:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
	
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("RU")		
		RU=RU+1.4*pixel_distance
	

	elif  pixels[i][0]==pixels[i][1]-1 and pixels[i][2]==pixels[i][3]+1:
	
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		if LU!=0:
			Commande.append(["LU",LU])
			LU=0
		#print ("LD")		
		LD=LD+1.4*pixel_distance

	elif  pixels[i][0]==pixels[i][1]-1 and pixels[i][2]==pixels[i][3]-1:
		if LD!=0:
			Commande.append(["LD",LD])
			LD=0
		if RU!=0:
			Commande.append(["RU",RU])
			RU=0
		if RD!=0:
			Commande.append(["RD",RD])
			RD=0
		if UU!=0:
			Commande.append(["UU",UU])
			UU=0
		if DD!=0:
			Commande.append(["DD",DD])
			DD=0
		if LL!=0:
			Commande.append(["LL",LL])
			LL=0
		if RR!=0:
			Commande.append(["RR",RR])
			RR=0
		
		#print ("LU")		
		LU=LU+1.4*pixel_distance
	
####################################################################""




def FROM(grid):
	global xs,ys,ii,jj
	destination = grid[xs][ys]
	ii=xs
	jj=ys

	destination.make_end()
	return destination

def vers(grid):
	global xf,yf
	destination = grid[xf][yf]

	destination.make_start()
	return destination
	




    
def obstacles1(grid):
	global TempBarrier
	for i in range (20):
		bar = grid[i][3]
		bar.make_barrier()
	for i in range (20):
		bar = grid[i][20]
		bar.make_barrier()
	for i in range (20):
		bar = grid[40][i]
		bar.make_barrier()
	for i in range (20):
		bar = grid[30][i]
		bar.make_barrier()
	for i in range(0,len(TempBarrier)):
		bar=grid[TempBarrier[i][0]][TempBarrier[i][1]]
		bar.make_barrier()
	
	
class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col
	def get_row(self):
		return self.row
	def get_col(self):
		return self.col

	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

	def make_path(self):
		self.color = PURPLE
		

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid, open=1):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])
		if self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and not grid[self.row + 1][self.col + 1].is_barrier(): #DOWNRIGHT
			self.neighbors.append(grid[self.row + 1][self.col + 1])
		if self.row < self.total_rows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier(): #DOWNLEFT
			self.neighbors.append(grid[self.row + 1][self.col - 1])
		if self.row > 0 and self.col < self.total_rows - 1 and not grid[self.row - 1][self.col + 1].is_barrier(): #UPRIGHT
			self.neighbors.append(grid[self.row - 1][self.col + 1])
		if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_barrier(): #UPLEFT
			self.neighbors.append(grid[self.row - 1][self.col - 1])
		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])
		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])
		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])

	def __lt__(self, other):
		return False


def h(p1, p2, c=0):
	x1, y1 = p1
	x2, y2 = p2
	if c == 0:
		return (abs(x1-x2) + abs(y1-y2)) * pixel_distance
	return math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)) * pixel_distance


def algorithm(draw, grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start] = h(start.get_pos(), end.get_pos())

	open_set_hash = {start}
	
	while not open_set.empty():
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end, draw,grid) 
			end.make_end()
			return True

		for neighbor in current.neighbors:
			
			xn, yn = neighbor.get_pos()
			xc, yc = current.get_pos()
			
			if xn == xc:
				temp_g_score = g_score[current] + pixel_distance
			elif yn == yc:
				temp_g_score = g_score[current] + pixel_distance
			else:
				temp_g_score = g_score[current] + pixel_distance * math.sqrt(2)

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()


		WIDTH

		if current != start:
			current.make_closed()

	return False


def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
	
		for j in range(rows):
			
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid


def draw_grid(win, rows, width):
	gap = width // rows

	for i in range(rows):
		pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
		for j in range(rows):
			
				
			pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))



def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)
	pygame.display.update()


def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap

	return row, col
def check(grid):
	for i in range (49):
		for j in range (49):
			if grid[i][j].color == PURPLE:
				print (i,j)

A=371
def obstacles(grid):
	for i in range (A):
		bar = grid[i][0]
		bar.make_barrier()
	for i in range (int(A*14.6/37.1)):
		bar = grid[0][i]
		bar.make_barrier()
	for i in range (int(A*0.7/37.1)+1): ##porte acceuil 1
		bar = grid[int(A*28.7/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*1.3/37.1),int(A*2/37.1)):
		bar = grid[int(A*28.7/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*14.6/37.1)):
		bar = grid[A][i]
		bar.make_barrier()
	for i in range (int(A*1.9/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*9.2/37.1)):
		bar = grid[int(A*1.9/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*1.9/37.1)):
		bar = grid[i][int(A*9.2/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*2.3/37.1)+1,int(A*5.4/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*6.5/37.1)):
		bar = grid[int(A*2.3/37.1)+1][i]
		bar.make_barrier()
	for i in range (int(A*6.7/37.1),int(A*8.5/37.1)):
		bar = grid[int(A*2.3/37.1)+1][i]
		bar.make_barrier()
	for i in range (int(A*8.7/37.1)+1,int(A*9.2/37.1)):
		bar = grid[int(A*2.3/37.1)+1][i]
		bar.make_barrier()
	for i in range (int(A*2.3/37.1)+1,int(A*5.4/37.1)):
		bar = grid[i][int(A*6.1/37.1)]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*9.2/37.1)):
		bar = grid[int(A*5.4/37.1)-1][i]
		bar.make_barrier()
	for i in range (A):
		bar = grid[i][int(A*14.6/37.1)]
		bar.make_barrier()
### escalier 
	for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
		bar = grid[int(A*3/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
		bar = grid[int(A*5.5/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*3/37.1),int(A*5.5/37.1)):
		bar = grid[i][int(A*0.6/37.1)]
		bar.make_barrier()
	for i in range (int(A*3/37.1),int(A*5.5/37.1)):
		bar = grid[i][int(A*1.3/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
		bar = grid[int(A*18.6/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
		bar = grid[int(A*22/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*18.6/37.1),int(A*22/37.1)):
		bar = grid[i][int(A*0.6/37.1)]
		bar.make_barrier()
	for i in range (int(A*18.6/37.1),int(A*22/37.1)):
		bar = grid[i][int(A*1.3/37.1)-1]
		bar.make_barrier()

	########## bloc C now 	
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*16.2/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*16.2/37.1),int(A*17.6/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*17.6/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*17.6/37.1),int(A*19.4/37.1)+1):
		bar = grid[i][int(A*12.7/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*19.4/37.1)+1][i]
		bar.make_barrier()
	for i in range (int(A*19.4/37.1)+1,int(A*21/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*21/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*18/37.1),int(A*19.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*18/37.1),int(A*19.1/37.1)):
		bar = grid[i][int(A*12.2/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*18/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*19.1/37.1)-1][i]
		bar.make_barrier()
		##
	for i in range (int(A*21/37.1),int(A*24.2/37.1)+1):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*21/37.1)-int(A*8.1/37.1),int(A*24.2/37.1)+1-int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*21/37.1)+int(A*8.1/37.1),int(A*24.2/37.1)+1+int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*21/37.1)-2*int(A*8.1/37.1),int(A*24.2/37.1)+1-2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	########## bloc B now 	
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*16.2/37.1)+int(A*8.1/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*16.2/37.1)+int(A*8.1/37.1),int(A*17.6/37.1)+int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*17.6/37.1)-1+int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*17.6/37.1)+int(A*8.1/37.1),int(A*19.4/37.1)+1+int(A*8.1/37.1)):
		bar = grid[i][int(A*12.7/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*19.4/37.1)+1+int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*19.4/37.1)+1+int(A*8.1/37.1),int(A*21/37.1)+int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*21/37.1)-1+int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*18/37.1)+int(A*8.1/37.1),int(A*19.1/37.1)+int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*18/37.1)+int(A*8.1/37.1),int(A*19.1/37.1)+int(A*8.1/37.1)):
		bar = grid[i][int(A*12.2/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*18/37.1)-1+int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*19.1/37.1)-1+int(A*8.1/37.1)][i]
		bar.make_barrier()
		
	########## bloc D now 	
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*16.2/37.1)-int(A*8.1/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*16.2/37.1)-int(A*8.1/37.1),int(A*17.6/37.1)-int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*17.6/37.1)-1-int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*17.6/37.1)-int(A*8.1/37.1),int(A*19.4/37.1)+1-int(A*8.1/37.1)):
		bar = grid[i][int(A*12.7/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*19.4/37.1)+1-int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*19.4/37.1)+1-int(A*8.1/37.1),int(A*21/37.1)-int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*21/37.1)-1-int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*18/37.1)-int(A*8.1/37.1),int(A*19.1/37.1)-int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*18/37.1)-int(A*8.1/37.1),int(A*19.1/37.1)-int(A*8.1/37.1)):
		bar = grid[i][int(A*12.2/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*18/37.1)-1-int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*19.1/37.1)-1-int(A*8.1/37.1)][i]
		bar.make_barrier()
	########## bloc E now 	
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*16.2/37.1)-2*int(A*8.1/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*16.2/37.1)-2*int(A*8.1/37.1),int(A*17.6/37.1)-2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*17.6/37.1)-1-2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*17.6/37.1)-2*int(A*8.1/37.1),int(A*19.4/37.1)+1-2*int(A*8.1/37.1)):
		bar = grid[i][int(A*12.7/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*19.4/37.1)+1-2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*19.4/37.1)+1-2*int(A*8.1/37.1),int(A*21/37.1)-2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*21/37.1)-1-2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*18/37.1)-2*int(A*8.1/37.1),int(A*19.1/37.1)-2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*18/37.1)-2*int(A*8.1/37.1),int(A*19.1/37.1)-2*int(A*8.1/37.1)):
		bar = grid[i][int(A*12.2/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*18/37.1)-1-2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*19.1/37.1)-1-2*int(A*8.1/37.1)][i]
		bar.make_barrier()
		#####
	for i in range (int(A*2.3/37.1)+1,int(A*3.4/37.1)):
		bar = grid[i][int(A*9.2/37.1)-1]
		bar.make_barrier()	
	for i in range (int(A*4.3/37.1),int(A*18/37.1)):
		bar = grid[i][int(A*9.2/37.1)-1]
		bar.make_barrier()	
	for i in range (int(A*5.3/37.1)+1,int(A*18/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()	
	for i in range (int(A*19.1/37.1),int(A*21.5/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()	
	for i in range (int(A*19.1/37.1),int(A*(21.5+11.8)/37.1)):
		bar = grid[i][int(A*9.2/37.1)-1]
		bar.make_barrier()	
	for i in range (int(A*2/37.1),int(A*9.1/37.1)):
		bar = grid[int(A*18/37.1)-1][i]
		bar.make_barrier()	
	for i in range (int(A*2/37.1),int(A*9.1/37.1)):
		bar = grid[int(A*19.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*9.1/37.1)):
		bar = grid[int(A*18.4/37.1)+1][i]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*9.1/37.1)):
		bar = grid[int(A*18.6/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*18.4/37.1)+1,int(A*18.6/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()
	for i in range (int(A*18.4/37.1)+1,int(A*18.6/37.1)):
		bar = grid[i][int(A*9.2/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*9.1/37.1)-1,int(A*9.4/37.1)): ##porte acceuil 
		bar = grid[int(A*(21.5+11)/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*9.8/37.1)-1,int(A*10/37.1)): ##porte acceuil 
		bar = grid[int(A*(21.5+11)/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*19.1/37.1),int(A*(21.5+11.8)/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*9.1/37.1)): ##porte acceuil 
		bar = grid[int(A*(21.5+11.8)/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*(21.5+11.8+1.3)/37.1),int(A*(21.5+11.8+1.3+2.7)/37.1)):
		bar = grid[i][int(A*2/37.1)]
		bar.make_barrier()
	for i in range (int(A*2/37.1),int(A*4.3/37.1)): ##porte acceuil 
		bar = grid[int(A*(21.5+11.8+1.3)/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*(21.5+11.8+1.3)/37.1),int(A*(21.5+11.8+1.3+2.7)/37.1)):
		bar = grid[i][int(A*4.3/37.1)]
		bar.make_barrier()
	########## bloc A now 	
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*16.2/37.1)+2*int(A*8.1/37.1)-1][i]
		bar.make_barrier()
	for i in range (int(A*16.2/37.1)+2*int(A*8.1/37.1),int(A*17.6/37.1)+2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*17.6/37.1)-1+2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*17.6/37.1)+2*int(A*8.1/37.1),int(A*19.4/37.1)+1+2*int(A*8.1/37.1)):
		bar = grid[i][int(A*12.7/37.1)-1]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.7/37.1)):
		bar = grid[int(A*19.4/37.1)+1+2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*19.4/37.1)+1+2*int(A*8.1/37.1),int(A*21/37.1)+2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*14.6/37.1)):
		bar = grid[int(A*21/37.1)-1+2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*18/37.1)+2*int(A*8.1/37.1),int(A*19.1/37.1)+2*int(A*8.1/37.1)):
		bar = grid[i][int(A*10/37.1)]
		bar.make_barrier()
	for i in range (int(A*18/37.1)+2*int(A*8.1/37.1),int(A*19.1/37.1)+2*int(A*8.1/37.1)):
		bar = grid[i][int(A*12.2/37.1)]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*18/37.1)-1+2*int(A*8.1/37.1)][i]
		bar.make_barrier()
	for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
		bar = grid[int(A*19.1/37.1)-1+2*int(A*8.1/37.1)][i]
		bar.make_barrier()
		

main(WIN, WIDTH)

