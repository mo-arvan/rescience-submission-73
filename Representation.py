import pygame
import math
import numpy as np 

UP,DOWN,LEFT,RIGHT,STAY=0,1,2,3,4

class World_Representation:

    def __init__(self, environment, table=np.zeros((0,0)),actions=np.zeros((0,0)),title='Best policy'):
        
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.size_coeff=50
        self.margin=5
        self.color = self.WHITE
        
        pygame.init()
        pygame.font.init()
        self.size = (5*self.size_coeff, 5*self.size_coeff)
        self.screen = pygame.Surface(self.size)
        
        self.font = pygame.font.SysFont('arial', 18)

        pygame.display.set_caption(title)    

        self.grid = np.zeros((5,5))
        self.init=(0,0)
        self.reward=(2,4)
        self.table=table
        self.actions=actions
        self.screen.fill(self.BLACK)

        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                self.color = self.WHITE
                pygame.draw.rect(self.screen,self.color,[(self.size_coeff)*col+self.margin,(self.size_coeff)*row+self.margin,
                                                                 self.size_coeff-self.margin,self.size_coeff-self.margin])
                
        
        #Gradient of the Q-values
        if table.size >0:
            for row in range(len(table)):
                for col in range(len(table[0])): 
                    self.color=(255,max(0,255*(1-table[(row,col)])),max(0,255*(1-table[row,col])))
                    pygame.draw.rect(self.screen,self.color,[(self.size_coeff)*col+self.margin,(self.size_coeff)*row+self.margin,
                                                                 self.size_coeff-self.margin,self.size_coeff-self.margin])
                    x,y=50*col+27.5,50*row+27.5
                    best_action=actions[row,col]
                    if best_action==STAY:
                        pygame.draw.circle(self.screen, self.BLACK, (x,y), 8)
                    if best_action!= STAY : 
                        angles_arrow=[0,180,270,90]
                        arrow_alignment={UP:(0,5),DOWN:(0,-5),LEFT:(5,0),RIGHT:(-5,0)}
                        x+=arrow_alignment[best_action][0]
                        y+=arrow_alignment[best_action][1]
                        self.DrawArrow(x, y,self.BLACK,angles_arrow[best_action])
        
        #Starting and goal state : S and G
        starting_state=self.font.render("S",1,self.BLACK)
        self.screen.blit(starting_state,(35,30))
        goal_state=self.font.render("G",1,self.BLACK)
        self.screen.blit(goal_state,(235,130))
                    

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
        

