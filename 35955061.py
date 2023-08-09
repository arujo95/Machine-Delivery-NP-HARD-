#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:20:12 2021

@author: arushijoshi
"""

import pandas as pd
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy

class vehicleRouting():
    """This class contains solution to the entire vehicle routing problem.First, it generates an initial solution, then operates on it to 
       find the optimal solution. It gives the user a choice between swap operator or insert operator to shuffle the customer between different 
       routes. Further, it also gives a choice between Hill Climbing and Simulated Annealing to find the optimal solution. Finally,
       It also generates a dynamic iteration Vs Cost graph and a best solution graph to give the user a beter understanding of
       the solution
    """
    
    def __init__(self, filename):      
        self.filename = filename
        self.file = pd.read_csv(filename)
        self.n = len(self.file['x'])
        self.file.set_index('ID', inplace = True)

    def InitialSolution(self):
        '''
        This function generates a random solution in form of list of list with every list contaning the customer ids which the particular
        truck will deliver to. The total number of list inside the outer list are n**0.5 or square root of n, and total number of 
        customers that each truck is delivering to is again n**0.5
        
        Returns
        -------
        self.sol: list of list
            This is the innitial solution generated.
    
        '''        
        # sol[] here is a list of list which contains lists of all the customers that all the trucks will visit
        self.sol=[]
        for i in range(0,self.n,int(self.n**0.5)):
            self.sol.append(list(range(i,int(self.n**0.5)+i)))
        np.random.shuffle(self.sol)
        return self.sol     
    
    
    def Cost(self, sol):
        '''
        This function calculates the cost or the sum of distance that all the truck travel and the delta. delta is the difference 
        between the largest distance travelled by a truck and the shortest distance travelled by a truck. Our objective is to 
        reduce this objective.
        
        Parameters
        ----------
        sol : list of list
            Cost of this solution is calculated.

        Returns
        -------
        int
            The objective generated This objective has to be further reduced.
        
        '''
      
        self.obj = 0
        #allDist is the list of distances travelled by trucks.
        self.allDist = []
        for i in range(int(self.n**0.5)):
            dist = 0
            for j in range(int(self.n**0.5)):
                if j == 0 :
                    dist += ((0 - self.file.iloc[sol[i][j],0])**2 + (0 -self.file.iloc[sol[i][j],1])**2)**0.5
                    dist += ((self.file.iloc[sol[i][j],0] - self.file.iloc[sol[i][j+1],0])**2 + (self.file.iloc[sol[i][j],1] - self.file.iloc[sol[i][j+1],1])**2)**0.5                           
                elif j == self.n**0.5-1:
                    dist += ((self.file.iloc[sol[i][j],0] - 0)**2 + (self.file.iloc[sol[i][j],1] - 0)**2)**0.5
                else:
                    dist += ((self.file.iloc[sol[i][j],0] - self.file.iloc[sol[i][j+1],0])**2 + (self.file.iloc[sol[i][j],1] - self.file.iloc[sol[i][j+1],1])**2)**0.5
            self.obj += dist
            self.allDist.append(dist)   
        delta = max(self.allDist) - min(self.allDist)
        self.obj += delta
        return self.obj
    
    
    def updatedCost(self, sol,x1,x2):
        '''
        This function recalculates the cost or the objective after swap operator has been performed on the solution. This is differnt
        from the cost function as this only calculates the distance of the trucks between which swap has been performed. This is supposed to
        increase the speed of finding the solution as it is only calculating the new distance of the truck between which the switch has
        occured.
        Parameters
        ----------
        sol : list of lists
            Still not the optimal solution.
        x1 : int
            1st truck.
        x2 : int
            2nd truck.

        Returns
        -------
        int
            sum of total distance travelled by all the trucks after the swap and dleta.

        '''
        
        self.obj_updated = 0
        for i in (self.x1, self.x2):
            dist = 0
            for j in range(int(self.n**0.5)):
                if j == 0 :
                    dist += ((0 - self.file.iloc[sol[i][j],0])**2 + (0 -self.file.iloc[sol[i][j],1])**2)**0.5
                    dist += ((self.file.iloc[sol[i][j],0] - self.file.iloc[sol[i][j+1],0])**2 + (self.file.iloc[sol[i][j],1] - self.file.iloc[sol[i][j+1],1])**2)**0.5                           
                elif j == self.n**0.5-1:
                    dist += ((self.file.iloc[sol[i][j],0] - 0)**2 + (self.file.iloc[sol[i][j],1] - 0)**2)**0.5
                else:
                    dist += ((self.file.iloc[sol[i][j],0] - self.file.iloc[sol[i][j+1],0])**2 + (self.file.iloc[sol[i][j],1] - self.file.iloc[sol[i][j+1],1])**2)**0.5
            self.obj_updated += dist
            self.allDist[i] = dist      
        return sum(self.allDist) + max(self.allDist) - min(self.allDist)

    
    def Feasibility(self,sol):
        '''
        This function calculates the feasibility of solution based on constraints. If the id of the 1st customer visited is odd or if 
        the id of the last customer visited is even, then it is a violation of constraint and thus the feasibility increases by 1
        for every such case.

        Parameters
        ----------
        sol : list of lists

        n : int
            total number of customers.

        Returns
        -------
        feas : int
            total number of violation of constraints.

        '''
        self.feas = 0
        for i in range(int(self.n**0.5)):   
            if (sol[i][0])%2 == 1:
                self.feas += 1           
            if (sol[i][-1])%2==0:
                self.feas += 1
        return self.feas
    
    def SwapOperator(self, sol, x1, y1, x2, y2):
        '''
        This function swaps the position of any two randomly selected customers.

        Parameters
        ----------
        sol : list of lists
        x1 : int
            randomly generated truck number.
        y1 : TYPE
            randomly generated customer position in sol.
        x2 : int
            randomly generated truck number.
        y2 : int
            randomly generated customer position in sol.

        Returns
        -------
        None.

        '''
        temp = self.sol[x1][y1]
        self.sol[x1][y1] = self.sol[x2][y2]
        self.sol[x2][y2] = temp
    
    def InsertOperator(self, sol, x1, y1, x2, y2):
        '''
        This is another function that operated on the solution. This results in not only the switch between any 2 elements
        of sol, but a shift in the entire structure of sol.

        Parameters
        ----------
        sol : list of lists
        x1 : int
            randomly generated truck number.
        y1 : int
            randomly generated customer position in sol.
        x2 : int
            randomly generated truck number.
        y2 : int
            randomly generated customer position in sol.

        Returns
        -------
        sol : list of lists
            new solution.

        '''
        sol_new = []
        temp = sol[x1][y1]
        sol[x1].pop(y1)
        sol[x2].insert(y2, temp)
        #print(sol)
        for i in range(int(self.n**0.5)):
            for j in range(len(sol[i])):
                sol_new.append(sol[i][j])
        
        sol = []
        for i in range(int(self.n**0.5)):
            x = []
            for j in range(int(self.n**0.5)):
                x.append(sol_new[0])
                sol_new.pop(0)
            sol.append(x)
        return sol
    
    
    def HillClimbing(self,sol, operator):
        '''
        This function gives us the optimal solution by rejecting the worsening moves. Both insert and swap operator can be used
        with this function.

        Parameters
        ----------
        sol : list of lsit

        Returns
        -------
        sol : list of lists
            optimal solution.

        '''
        # variables for later generating dynamic graph
        self.iteration_no = []
        self.costPerIter = []
        
        #iteration is the number of times the operation will be performed on the initial solution
        self.iterations = int(input('Enter the number of iterations      : ')) 
        for i in range(self.iterations):
            # generating randon numbers upto length of the solution i.e. n to select the elements of swap and insert operation 
            self.x1 = random.randint(0, len(sol)-1)
            self.y1 = random.randint(0, len(sol)-1)
            self.x2 = random.randint(0, len(sol)-1)
            self.y2 = random.randint(0, len(sol)-1)
            old_sol = copy.deepcopy(sol)
            
            if operator == 1:
                #for swap operator
                self.solCost = self.Cost(sol)
                self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
                self.solCost_new = self.updatedCost(sol, self.x1, self.x2)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
            elif operator == 2:
                #for insert operator
                self.solCost = self.Cost(sol)
                self.sol = copy.deepcopy(self.InsertOperator(sol, self.x1, self.y1, self.x2, self.y2))
                self.solCost_new = self.Cost(self.sol)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(self.sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    self.sol = copy.deepcopy(old_sol)   
                sol = copy.deepcopy(self.sol)
                
              # with swap or insert operator  
            elif operator == 3 and np.random.randint(1,3) == 1:
                #for swap operator
                self.solCost = self.Cost(sol)
                self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
                self.solCost_new = self.updatedCost(sol, self.x1, self.x2)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
            elif operator == 3 and np.random.randint(1,3) == 2:
                #for insert operator
                self.solCost = self.Cost(sol)
                self.sol = copy.deepcopy(self.InsertOperator(sol, self.x1, self.y1, self.x2, self.y2))
                self.solCost_new = self.Cost(self.sol)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(self.sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    self.sol = copy.deepcopy(old_sol) 
                sol = copy.deepcopy(self.sol)
            self.iteration_no.append(i)    
            self.costPerIter.append(self.solCost)    
                
        return sol
        
    def SimulatedAnnealing(self, sol, operator):
        '''
        This function gives us an optimal solution by accepting worsening moves by a certain probability. Both swap operator and
        insert operator can be sued with this function. 

        Parameters
        ----------
        sol : list of lists
    

        Returns
        -------
        sol : list of lists
            This is the optimal solution
        '''
        # variables for later generating dynamic graph
        self.iteration_no = []
        self.costPerIter = []
        
        #iteration is the number of times the operation will be performed on the initial solution
        self.iterations = int(input('Enter the number of iterations      : ')) 
        
        #as the temperaturre decreases exponentially, we start getting an optimal solution
        Tc = int(input('Enter the temperature               : '))
        
        #rate at which temperature decreases
        mult = float(input('Enter multiplier between 0 and 1    : '))
        self.solCost = self.Cost(sol)
        
        
        for i in range(self.iterations):
            temp = Tc * mult
            # generating randon numbers upto length of the solution i.e. n to select the elements of swap and insert operation
            self.x1 = random.randint(0, len(sol)-1)
            self.y1 = random.randint(0, len(sol)-1)
            self.x2 = random.randint(0, len(sol)-1)
            self.y2 = random.randint(0, len(sol)-1)
            old_sol = copy.deepcopy(self.sol)
            
            if operator ==1:
                # with swap operator
                self.solCost = self.Cost(sol)
                self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
                self.solCost_new = self.updatedCost(sol,self.x1,self.x2)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    w = np.exp(-(self.solCost_new - self.solCost)/Tc)
                    s = np.random.random()
                    if w>s:
                        self.solCost = self.solCost_new
                    else:
                        self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2) 
                
            elif operator == 2:
                # with insert operator
                old_sol = copy.deepcopy(self.sol)#innitial soln
                self.solCost = self.Cost(self.sol)
                self.sol = copy.deepcopy(self.InsertOperator(sol, self.x1, self.y1, self.x2, self.y2))
                self.solCost_new = self.Cost(self.sol)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(self.sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    w = np.exp(-(self.solCost_new - self.solCost)/Tc)
                    s = np.random.random()
                    if w>s:
                        self.solCost = self.solCost_new
                    else:
                        self.sol = copy.deepcopy(old_sol)
                sol = copy.deepcopy(self.sol)
                
             #with swap or insert operator   
            elif operator ==3 and np.random.randint(1,3) ==1:
                # with swap operator
                self.solCost = self.Cost(sol)
                self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
                self.solCost_new = self.updatedCost(sol,self.x1,self.x2)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    w = np.exp(-(self.solCost_new - self.solCost)/Tc)
                    s = np.random.random()
                    if w>s:
                        self.solCost = self.solCost_new
                    else:
                        self.SwapOperator(sol, self.x1, self.y1, self.x2, self.y2)
            elif operator ==3 and np.random.randint(1,3) ==2:
                  # with insert operator
                old_sol = copy.deepcopy(self.sol)#innitial soln
                self.solCost = self.Cost(self.sol)
                self.sol = copy.deepcopy(self.InsertOperator(sol, self.x1, self.y1, self.x2, self.y2))
                self.solCost_new = self.Cost(self.sol)
                if self.solCost_new < self.solCost:
                    self.solCost = self.solCost_new
                elif self.solCost_new == self.solCost and self.Feasibility(self.sol) < self.Feasibility(old_sol):
                    self.solCost = self.solCost_new
                else:
                    w = np.exp(-(self.solCost_new - self.solCost)/Tc)
                    s = np.random.random()
                    if w>s:
                        self.solCost = self.solCost_new
                    else:
                        self.sol = copy.deepcopy(old_sol)
                sol = copy.deepcopy(self.sol)
            self.iteration_no.append(i)    
            self.costPerIter.append(self.solCost) 
            Tc = temp
        return sol
    
         
    def iterationVsCost(self):
        '''
        This is the iteration Vs Cost line graph. It shows how the Cost decreases with every iteration. This is 
        the objective. 

        Returns
        -------
        None.

        '''
        iter_100 = []
        cost_100 = []
        for j in range(self.iterations):
            iter_100.append(int(self.iteration_no[j]))
            cost_100.append(float(self.costPerIter[j]))
            if j%100 ==0:      
                fig, ax = plt.subplots()
                plt.plot(iter_100, cost_100)
                plt.xlabel("iteration number")
                plt.ylabel("cost")
                plt.title("Cost Vs Iterations")
       
    
    def bestSolGraph(self,sol):
        '''
        This functions plots a graph of the routes of all the delivery vehicles

        Parameters
        ----------
        sol : list of list
            here, sol is the optimal solution

        Returns
        -------
        None.

        '''
        vehicle_no = 0
        #generating list of x and y coordinates of the routes of all the trucks
        for i in range(int(self.n**0.5)):
            x_axis=[0,0]
            y_axis=[0,0]
            for j in range(int(self.n**0.5)):
                x = int(self.file.iloc[sol[i][j],0])
                y = int(self.file.iloc[sol[i][j],1])
                x_axis.insert(-1,x)
                y_axis.insert(-1,y)
            vehicle_no += 1
            #plotting line graph of all the routes
            plt.plot(x_axis,y_axis, label = "Vehicle {}".format(vehicle_no))
            #plotting scatter plot of all the customer positions
            plt.scatter(x_axis, y_axis)
            plt.title("Best Solution")
            plt.legend()
        plt.grid(True)
    
    
    def sol_Csv(self, sol):
        '''
        This function creates a csv in the format asked.

        Parameters
        ----------
        sol : list of lists
            Sol here is the optimal solution.

        Returns
        -------
        None.

        '''
        with open('/Users/arushijoshi/Desktop/MSCI530/sol.csv', 'w') as writeFile:
            # For each truck, the solution route is printed on a different line
            for i in range(int(self.n**0.5)):
                writeFile.write(','.join(str(s) for s in sol[i]))
                writeFile.write('\n')
            # Cost is saved in the next line
            writeFile.write(str(round(self.solCost,2)))
            writeFile.write('\n')
            #Feasibility is saved in the next line
            writeFile.write(str(self.Feasibility(sol)))
            
       
    def pickle_log(self, sol):
        '''
        This function creates a log file of all the previous solutions.

        Parameters
        ----------
        sol : list of lists
            this here is the optimal solution

        Returns
        -------
        None.

        '''
        pickledict = {'File name: ': self.filename, 'Initial Solution: ' : self.InitialSolution(), 'Initial Solution Cost: ': self.Cost(self.InitialSolution()), 'Initial Solution Feasibility: ': self.Feasibility(self.InitialSolution()), 'Final Solution: ': sol, 'Final Solution Cost: ': self.Cost(sol), 'Final Solution Feasibility: ': self.Feasibility(sol)}
        #open the pickle file to read
        try :
            picklefile = open("record.vr","rb")
            prev_sol = pickle.load(picklefile)
            print("Previous solution", prev_sol)
            picklefile.close()
        except FileNotFoundError :
            print("File does not exist")
        
        #updating the log file
        picklefile = open("record.vr","ab")
        pickle.dump(pickledict,picklefile)
        picklefile.close()
        

    
class Activate():     
    '''This is the main class where all the functions from the vehicleRouting class are called.'''
    
    def activate(self):
        #Asking the user for the file name.
        filename = input("Enter the file name: ")
        print(filename)
        
        #calling the vehicleRouting class with filename as the parameter
        self.vrp = vehicleRouting(filename)
        
        
        print("\n\n")
        #asking user for option between Hill Climbing or Simulated Annealing
        print("For Hill CLimbing enter             : 1 \nFor Simulated Annealing enter       : 2")
        algo = int(input("Enter the option                    : "))
        print("\n\n")
        #Asking user for option between swap operator, insert operator, or both
        print("For Swap Operator enter             : 1 \nFor Insert Operator Enter           : 2 \nFor Swap and Insert Operator enter  : 3 ")
        operator = int(input("Enter the option                    : "))
        print("\n\n")
        
        #Calling Hill Climbing or Simulated Annealing function depending or the user's response
        if algo==1:
            sol = copy.deepcopy(self.vrp.HillClimbing(self.vrp.InitialSolution(), operator))
        elif algo == 2:
            sol = copy.deepcopy(self.vrp.SimulatedAnnealing(self.vrp.InitialSolution(), operator))
        
        #printing the results
        print("\n\n")    
        print("Best Solution:\n",sol)
        print("\n")
        print("Best Solution Cost                  : ",self.vrp.solCost)
        print("\n")
        print("Best Solution Feasibility           ; ",self.vrp.Feasibility(sol))
        print("\n")
        print("-*-*-*-*-*-*-*-*Run Completed Successfully*-*-*-*-*-*-*-*-")
        print("\n\n")
        
        #calling the function to update the log
        self.vrp.pickle_log(sol)
        #Calling the function to update the CSV file
        self.vrp.sol_Csv(sol)
        #calling the function to plot the best solution graph
        plt.figure()
        self.vrp.bestSolGraph(sol)
        plt.pause(1)
        #calling the function to plot the cost vs iteration graph
        plt.figure()
        self.vrp.iterationVsCost()
        print("Log File:\n")
        

run = Activate()
run.activate()
