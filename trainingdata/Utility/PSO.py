from Particle import Particle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ParticleSwarm(object):
   
    def __init__(self, swarmSize, dimensions, minVal, maxVal):
        self.maxVal = maxVal
        self.minVal = minVal
        self.dimensions = dimensions
        self.swarmSize = swarmSize
        self.swarm = []
        self.solution = []
        for i in range(swarmSize):
            particle = Particle(dimensions, minVal, maxVal)
            self.swarm.append(particle)

        print(self.swarm)
        
        return


    def run(self, steps):
        
        self.swarm = sorted(self.swarm)
        self.ThreeDplotParticles(0)
        
        for step in range(steps):
            print("---Itteration " + str(step) + "---")
            
              
            globalBest = self.swarm[0]
            self.solution = globalBest
            
            
            for particle in self.swarm:
                particle.updateParticle(globalBest.getPositionList())
                
            self.swarm = sorted(self.swarm)
            
            #Print swarm
            for particle in self.swarm:     
                print(particle)
                
            self.ThreeDplotParticles(step+1)
            
    def ThreeDplotParticles(self, step):     
        xlist = []
        ylist = []
        zlist = []
        for particle in self.swarm:
            posList = particle.getPositionList()
            xlist.append(posList[0])
            ylist.append(posList[1])
            zlist.append(posList[2])
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xlist, ylist, zlist, c='b', marker='o')
        
        ax.set_xlabel('X Value')
        ax.set_ylabel('Y Value')
        ax.set_zlabel('Z Value')
        
        ax.set_xlim3d(0, 20)
        ax.set_ylim3d(0,20)
        ax.set_zlim3d(0,20)
                            
        plt.title("Swarm at Iteration " + str(step))
        plt.legend(loc='upper right')
        plt.savefig('image/'+str(step)+'.png') 
        plt.close()  
         
    def TwoDplotParticles(self, step):     
        xlist = []
        ylist = []
        for particle in self.swarm:
            posList = particle.getPositionList()
            xlist.append(posList[0])
            ylist.append(posList[1])
            
        plt.scatter(x=xlist, y=ylist, label='Particles', color = 'blue', )
                
        plt.title("Swarm at Iteration " + str(step))
        plt.xlabel('X Val')
        plt.ylabel('Y Val')
        plt.axis((0,20,0,20))
        plt.legend(loc='upper right')
        plt.savefig('image/'+str(step)+'.png') 
        plt.close()  
            
            
            
            #for particle in self.swarm:
            #    if(particle)
                


def main():
    swarmSize = 500
    dimensions = 3
    minVal = 0
    maxVal = 20
    swarm = ParticleSwarm(swarmSize, dimensions, minVal, maxVal)
    print(swarm.run(50))








if __name__ == '__main__':
    main() 
