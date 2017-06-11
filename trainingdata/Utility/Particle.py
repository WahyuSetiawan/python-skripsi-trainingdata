import random
from sklearn import svm

class Particle(object):
    def __init__(self, particle, index, svm, label, weight):
        self.index = index
        self.dimensions = len(particle)
        self.position = []
        self.velocity = []
        self.selfBest = []
        self.fitness = 0
        self.bestFitness = 0
        self.svm = svm
        self.label = label
        self.weight = weight
        for i in range(self.dimensions):
            self.position.append(particle[i])
            self.velocity.append(random.random())
            self.selfBest.append(self.position[i])
            
        self.updatefitnessfunction()
        self.bestFitness = self.fitness
        return
        
    def __repr__(self):
        val = ""
        for i in range(len(self.position)):
            val += "%.2f" % self.position[i] + ", "
            
        vol = ""
        for i in range(len(self.velocity)):
            vol += "%.2f" % self.velocity[i] + ", "
            
        return "Fitness:" + "%.2f" % self.fitness + " (" + val[:len(val)-2] + ")" + " (" + vol[:len(vol)-2] + ")"
           
    def getPositionList(self):
        return self.position        
           
    def updateParticle(self, globalBest, weight):
        self.updateVelocity(globalBest, weight)
        self.updatePosition()
        self.updatefitnessfunction()
        return
           
    def updatePosition(self):
        for i in range(self.dimensions):
            self.position[i] += self.velocity[i]
        return
            
    def updateVelocity(self, globalBest, weight):
        for i in range(self.dimensions):
            #Get random Vals
            localRandom = random.randint(0,1)
            globalRandom = random.randint(0,1)
            
            #Get the subtractions of lacal and global
            localVal = 2 * localRandom * (self.selfBest[i] - self.position[i])
            globalVal = 2 * globalRandom * (globalBest[i] - self.position[i])
            
            #Update the velocity
            self.velocity[i] = (self.weight[i] * self.velocity[i]) + localVal + globalVal
            

    def updatefitnessfunction(self):
        fitness = self.svm.decision_function(self.position)[0][self.label]

        if(self.bestFitness > fitness):
            self.selfBest = self.position
            self.bestFitness = fitness
        self.fitness = fitness

        '''
        fitness = 0
        for i in range(self.dimensions):
            fitness += abs(10 - self.position[i])
            
        if(self.bestFitness > fitness):
            self.selfBest = self.position
            self.bestFitness = fitness
        self.fitness = fitness
        #'''

        return
        
    def __lt__(self, other):
        return (self.fitness < other.fitness)      
    def __gt__(self, other):
        return (self.fitness > other.fitness) 
    def __eq__(self, other):
        return (self.fitness == other.fitness) 
    def __le__(self, other):
        return (self.fitness <= other.fitness) 
    def __ge__(self, other):
        return (self.fitness >= other.fitness) 
    def __ne__(self, other):
        return (self.fitness != other.fitness) 
    