import pygame
import math
import numpy as np 

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

class World_Representation:

    def __init__(self, environment, table=np.zeros((0,0)),actions={},title='Best policy'):
        
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.WIDTH = 45
        self.HEIGHT = 45
        self.MARGIN = 5
        self.color = self.WHITE
        
        pygame.init()
        pygame.font.init()
        self.size = (environment.height*50, environment.width*50)
        self.screen = pygame.Surface(self.size)
        
        self.font = pygame.font.SysFont('arial', 18)

        pygame.display.set_caption(title)    

        self.grid = environment.grid
        self.init=(environment.first_location[0],environment.first_location[1])
        self.reward=(2,4)
        self.table=table
        self.actions=actions
        self.screen.fill(self.BLACK)

        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                self.color = self.WHITE
                pygame.draw.rect(self.screen,self.color,
					[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					(self.MARGIN + self.HEIGHT)*row+self.MARGIN,
					self.WIDTH,self.HEIGHT])
                
        
        
        if table.size >0:
            for row in range(len(table)):
                for col in range(len(table[0])): 
                    if self.grid[row][col]!=-1:
                        self.color=(255,max(0,255*(1-table[(row,col)])),max(0,255*(1-table[row,col])))
                        pygame.draw.rect(self.screen,
					                 self.color,[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					                 (self.MARGIN + self.HEIGHT)*row+self.MARGIN,self.WIDTH,self.HEIGHT])
                         
        if actions!={}:
            for (row,col),value in actions.items():
                    if self.grid[row][col]!=-1:
                        best_action=value
                        x,y=50*col+27.5,50*row+27.5
                        if best_action==STAY:
                            pygame.draw.circle(self.screen, self.BLACK, (x,y), 8)
                        if best_action != STAY : 
                            angles={UP:0,DOWN:180,RIGHT:270,LEFT:90}
                            correction={UP:(0,5),DOWN:(0,-5),LEFT:(5,0),RIGHT:(-5,0)}
                            x+=correction[best_action][0]
                            y+=correction[best_action][1]
                            self.DrawArrow(x, y,self.BLACK,angles[best_action])
       
        starting_state=self.font.render("S",1,self.BLACK)
        self.screen.blit(starting_state,(35,30))
        goal_state=self.font.render("G",1,self.BLACK)
        self.screen.blit(goal_state,(35+4*50,30+2*50))
                    

    def DrawArrow(self,x,y,color,angle=0):
        def rotate(pos, angle):	
            cen = (x,y)
            angle *= -(math.pi/180)
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)
            ret = ((cos_theta * (pos[0] - cen[0]) - sin_theta * (pos[1] - cen[1])) + cen[0],
                   (sin_theta * (pos[0] - cen[0]) + cos_theta * (pos[1] - cen[1])) + cen[1])
            return ret		
        p0 = rotate((0+x,-8+y), angle+90)
        p1 = rotate((0+x,8+y), angle+90)
        p2 = rotate((16+x,0+y), angle+90)
        pygame.draw.polygon(self.screen, color, [p0,p1,p2])



#Enables to see the transitions of a given world

class Transition():

    def __init__(self, init, titre,transitions,walls,action,screen_size=450,cell_width=130, cell_height=130, cell_margin=15):
        self.WHITE=(255,255,255)
        self.BLACK=(0,0,0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.WIDTH = cell_width
        self.HEIGHT = cell_height
        self.MARGIN = cell_margin

        pygame.init()
        pygame.font.init()
        self.size = (screen_size, screen_size)
        self.screen = pygame.display.set_mode(self.size)

        self.font = pygame.font.SysFont('arial', 20)

        pygame.display.set_caption(titre)    

        self.transitions=transitions
        self.walls=walls
        self.screen.fill(self.BLACK)
        self.init=init

        for row in range(3):
            for col in range(3):
                if walls[(row+self.init[0]-1,col+self.init[1]-1)] == 1:
                    self.color =self.BLACK
                else:
                    self.color = self.WHITE
                pygame.draw.rect(self.screen,
					self.color,
					[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					(self.MARGIN + self.HEIGHT)*row+self.MARGIN,
					self.WIDTH,
					self.HEIGHT])
                y=150*row+45
                x=150*col+50
                x_row, y_col =row+self.init[0]-1, col+self.init[1]-1
                if (x_row,y_col) in self.transitions.keys():
                    label=self.font.render(str(round(self.transitions[(x_row,y_col)]*100,3))+" %",1,self.BLACK)
                    self.screen.blit(label,(x,y))
                if (row,col)==(1,1):
                        position={UP:(0,0),DOWN:(0,-35),LEFT:(10,-20),RIGHT:(-10,-20)}
                        x=220+position[action][0]
                        y=275+position[action][1]
                        angles={UP:0,DOWN:180,RIGHT:270,LEFT:90}
                        angle=angles[action]
                        self.DrawArrow(x, y,self.BLUE,angle)
    def DrawArrow(self,x,y,color,angle=0):
        def rotate(pos, angle):	
            cen = (5+x,0+y)
            angle *= -(math.pi/180)
            cos_theta = math.cos(angle)
            sin_theta = math.sin(angle)
            ret = ((cos_theta * (pos[0] - cen[0]) - sin_theta * (pos[1] - cen[1])) + cen[0],
                   (sin_theta * (pos[0] - cen[0]) + cos_theta * (pos[1] - cen[1])) + cen[1])
            return ret		
        p0 = rotate((0+x,-20+y), angle+90)
        p1 = rotate((0+x,20+y), angle+90)
        p2 = rotate((40+x,0+y), angle+90)
        pygame.draw.polygon(self.screen, color, [p0,p1,p2])



def show_transition(world_number,action,row,col):
    world=np.load('Environments/Transitions_' + str(world_number) +'.npy')
    all_transitions=np.load('Environments/Transitions_'+str(world_number)+'.npy',allow_pickle=True)
    transition=all_transitions[action][row][col]
    walls={(row-1,col-1):0,(row-1,col):0,(row-1,col+1):0,(row,col-1):0,(row,col):0,(row,col+1):0,(row+1,col-1):0,(row+1,col):0,(row+1,col+1):0}
    for case in walls.keys() : 
        if case[0]<0 or case[1]<0 or case[0]>=len(world) or case[1]>=len(world) or world[case[0],case[1]]==-1:
            walls[case]=1
    actions=['UP','DOWN','LEFT','RIGHT','STAY']
    titre ="Action " +actions[action]+ ", case ("+str(row)+","+str(col)+")"+" dans le monde "+str(world_number)
    transi=Transition((row,col),titre,transition,walls,action)
    pygame.display.flip()
    pygame.image.save(transi.screen,"Environments/"+str(titre)+'.png')
    pygame.time.delay(5000)
    pygame.quit()