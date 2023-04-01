import numpy as np


Glider1 = np.array([[0,0,1],[1,0,1],[0,1,1]])
Glider2 = np.array([[0,1,0],[1,0,0],[1,1,1]])
Glider3 = np.array([[1,1,0],[1,0,1],[1,0,0]])
Glider4 = np.array([[1,1,1],[0,0,1],[0,1,0]])

R_Pentimino1 = np.array([[0,1,1],[1,1,0],[0,1,0]])
R_Pentimino2 = np.array([[0,1,0],[1,1,1],[0,0,1]])
R_Pentimino3 = np.array([[0,1,0],[0,1,1],[1,1,0]])
R_Pentimino4 = np.array([[1,0,0],[1,1,1],[0,1,0]])

Exploder1 = np.array([[1,0,1],[1,0,1],[1,1,1]])
Exploder2 = np.array([[1,1,1],[1,0,0],[1,1,1]])
Exploder3 = np.array([[1,1,1],[1,0,1],[1,0,1]])
Exploder4 = np.array([[1,1,1],[0,0,1],[1,1,1]])

CompareList = [Glider1,Glider2,Glider3,Glider4,R_Pentimino1,R_Pentimino2,R_Pentimino3,R_Pentimino4,Exploder1,Exploder2,Exploder3,Exploder4]

def evaluate(cells: np.ndarray):
    
	xSize = cells.shape[0]
	ySize = cells.shape[1]
	fitness = 0
	
	for y in range(ySize):
		if  y == 0 or y== ySize -1: continue #Ignore top and bottom blocks
		for x in range(xSize):
			if x == 0 or x == xSize-1: continue #Ignore edge blocks
			
			#Create 3x3 matrix array for current position
			Target = np.array( [[cells[x-1,y-1],	cells[x,y-1],	cells[x+1,y-1]],
								[cells[x-1,y],		cells[x,y],		cells[x+1,y]],
								[cells[x-1,y-1],	cells[x,y-1],	cells[x+1,y-1]]])
			
			for Objects in CompareList:
				match = numpy.array_equal(Target,Object)
				if match:
					fitness += 1
					break
			
			
			
				
