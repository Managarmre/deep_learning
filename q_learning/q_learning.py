# @author Pauline Houlgatte
# based on http://mnemstudio.org/path-finding-q-learning-tutorial.htm


import random
import copy



# YOU CAN MODIFY T, alpha and gamma


# connection between two rooms
# -1 = wall / 0 = door (connection between two states) / 100 = goal
# 
# T = [[-1,-1,-1,-1,0,-1],
# 	 [-1,-1,-1,0,-1,0],
# 	 [-1,-1,-1,0,-1,-1],
# 	 [-1,0,0,-1,0,-1],
# 	 [0,-1,-1,0,-1,0],
# 	 [-1,0,-1,-1,0,-1]]


 # map representation (X = Wall)
 # XXXXXXXXXXXXXXXXXXXXXXXXXXX
 # X             X           X
 # X             X           X        5
 # X      0      X      1    
 # X             X           X
 # X             X           X
 # XXXXXXXX   XXXXXXXX    XXXXXXXXXXXXXXXXXXXXXXXX
 # X             X                X              X
 # X             X                X              X
 # X             X                X              X
 # X      4            3                  2      X
 # X             X                               X
 # X             X                X              X
 # X             X                X              X
 # X    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 #

# autre matrice de transition fonctionnelle
T = [[-1,0,-1,0,-1,-1,-1,-1,-1],
	 [0,-1,-1,0,-1,-1,-1,-1,-1],
	 [-1,-1,-1,-1,0,-1,-1,-1,-1],
	 [0,0,-1,-1,-1,0,0,-1,-1],
	 [-1,-1,0,-1,-1,-1,-1,-1,0],
	 [-1,-1,-1,0,-1,-1,-1,-1,-1],
	 [-1,-1,-1,0,-1,-1,-1,0,-1],
	 [-1,-1,-1,-1,-1,-1,0,-1,0],
	 [-1,-1,-1,-1,0,-1,-1,0,-1]]

# map representation
#	######################################################
#	#                     #                #             #
#	#                     #                #             #
#	#           0                   1      #     2       #
#	#                     #                #             #
#	#                     #                #             #
#	#                     #                #             #
#	########             ##########     ###########    ###
#	#                                       #            #
#	#                                       #            #
#	#                                       #            #
#	#                    3                  #      4     #
#	#                                       #            #
#	#                                       #            #
#	#                                       #            #
#	#                                       #            #
#	#                                       #            #
#	#                                       #            #
#	#####  ##################       #############    #####
#	#            #                  #           #        #
#	#            #                  #           #        #
#	#            #                  #           #        #
#	#            #              6          7        8    #
#	#      5     #                  #           #        #
#	#            #                  #           #        #
#	#            #                  #           #        #
#	#            #                  #           #        #
#	######################################################

# param
alpha = 0.8
gamma = 0.4

#################################

# Q
Q = [[0. for i in range (len(T[0]))] for j in range (len(T))]

def spawnAgent(position):
	spawn = random.randint(0,len(T)-1)
	while spawn == position:
		spawn = random.randint(0,len(T[0])-1)
	return spawn

def rewardMatrix(position):
	R = [[0 for i in range (len(T[0]))] for j in range (len(T))]
	for line in range(len(T)):
		for element in range(len(T[0])):
			if T[line][element] == -1:
				R[line][element] = -1
			elif element == position:
				R[line][element] = 100
			else:
				R[line][element] = 0
	return R

def updateQ(state,action,next_state,reward):
	q_s_a = Q[state][action]
	r_s_next = reward[state][action]

	q = q_s_a + alpha * (r_s_next + gamma * max(Q[next_state]) - q_s_a )
	Q[state][action] = q

def select_action(state):
	action = random.randint(0,len(T)-1)
	while T[state][action] == -1:
		action = random.randint(0,len(T[0])-1)
	return action

def best_way(robot,human):
	way = [robot]
	while human not in way:
		best_action = max(Q[way[len(way)-1]])
		next_state = Q[way[len(way)-1]].index(best_action)
		way.append(next_state)

	return way

def main():
	print "===================== ",0," ====================="
	# print T
	# print R
	robot = spawnAgent(-1)
	human = spawnAgent(robot)
	print "start robot :",robot, "\nstart human : ", human
	R = rewardMatrix(human)
	# print R

	test = 1000
	while test != 0:
		state = random.randint(0,len(T)-1)
		action = select_action(state)
		updateQ(state,action,action,R)

		test -= 1

	# i = 0
	# for line in Q:
	# 	print i," : ",line
	# 	i += 1

	print best_way(robot,human)

	for i in range(10):
		print "===================== ",i+1," ====================="
		robot = copy.deepcopy(human)
		human = spawnAgent(robot)
		print "start robot :",robot, "\nstart human : ", human
		R = rewardMatrix(human)

		test = 1000
		while test != 0:
			state = random.randint(0,len(T)-1)
			action = select_action(state)
			updateQ(state,action,action,R)

			test -= 1

		# i = 0
		# for line in Q:
		# 	print i," : ",line
		# 	i += 1

		print best_way(robot,human)

if __name__ == '__main__':
	main()